"""Compare Stage 0..3 on a common gallery/query manifest and occlusion bank."""
import _bootstrap
import argparse
import hashlib
import json
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from progressive.checkpoints import trusted_load
from progressive.evaluation import encoder_from_checkpoint, fixed_input, extract


def extract_rows(model, rows, root, device, name, parsing_root=None, mode='clean', mask_bank=None,
                 fake_model=None):
    features = []
    logging.info('Extracting %s: %d images', name, len(rows))
    for index, row in enumerate(rows, 1):
        tensor, _ = fixed_input(root / row['path'], root, parsing_root, mode, mask_bank)
        encoder = fake_model if fake_model is not None and 'target_id' in row else model
        features.append(extract(encoder, tensor, device))
        if index % 256 == 0 or index == len(rows):
            logging.info('%s: %d/%d', name, index, len(rows))
    return features


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--manifest',required=True,help='JSON with gallery and queries lists')
    p.add_argument('--data_root',required=True)
    p.add_argument('--parsing_root')
    p.add_argument('--mask_bank')
    p.add_argument('--modes',nargs='+',default=['clean','single','double','triple','inner'],
                   choices=['clean','single','double','triple','inner'])
    p.add_argument('--device',default='cuda')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix('.log')
    logging.basicConfig(level=logging.INFO, format='Testing: %(asctime)s-%(message)s',
                        handlers=[logging.FileHandler(log_path, mode='w', encoding='utf-8')])
    logging.info('Manifest: %s   Device: %s', args.manifest, args.device)
    torch.set_num_threads(2)
    raw=Path(args.manifest).read_bytes(); manifest=json.loads(raw)
    gallery,queries=manifest['gallery'],manifest['queries']
    if not gallery or not queries:
        raise ValueError('Gallery and queries must be nonempty')
    root=Path(args.data_root).resolve()
    gallery_paths={str((root/r['path']).resolve()) for r in gallery}
    if any(str((root/q['path']).resolve()) in gallery_paths for q in queries):
        raise ValueError('Gallery and query may not contain the exact same image')
    # Reject duplicate bytes even when files have different names.
    gallery_hashes={hashlib.sha256((root/r['path']).read_bytes()).digest() for r in gallery}
    if any(hashlib.sha256((root/q['path']).read_bytes()).digest() in gallery_hashes for q in queries):
        raise ValueError('Duplicate image content in gallery and queries')
    identities=sorted({str(r['identity']) for r in gallery})
    is_fake = torch.tensor(['target_id' in query for query in queries], dtype=torch.bool)
    if len(identities)<2:
        raise ValueError('Gallery must contain at least two identities')
    for q in queries:
        target=str(q.get('target_id',q.get('identity')))
        if target not in identities or ('source_id' in q and str(q['source_id']) not in identities):
            raise ValueError('Gallery must contain query Target and Source identities')
        if 'source_id' in q and str(q['source_id'])==target:
            raise ValueError('Fake Target and Source must be different')
    report={'manifest_sha256':hashlib.sha256(raw).hexdigest(),
            'note':'Dual-encoder Stage 2 uses its Real encoder for gallery and Real queries, and Student for Fake.',
            'checkpoints':{}}
    for checkpoint in args.checkpoints:
        logging.info('Checkpoint: %s', checkpoint)
        state = trusted_load(checkpoint)
        model = encoder_from_checkpoint(state, args.device)
        fake_model = encoder_from_checkpoint(state, args.device, role='student') if 'real_encoder' in state else None
        del state
        gfeats = extract_rows(model, gallery, root, args.device, 'Gallery')
        clean = extract_rows(model, queries, root, args.device, 'Queries (clean)', fake_model=fake_model)
        result={}
        for mode in args.modes:
            qfeats = clean if mode == 'clean' else extract_rows(
                model, queries, root, args.device, f'Queries ({mode})',
                args.parsing_root, mode, args.mask_bank, fake_model)
            result[mode]={}
            for feature in ('cls','patch_mean'):
                centers=torch.stack([F.normalize(torch.stack([gfeats[i][feature] for i,r in enumerate(gallery)
                           if str(r['identity'])==identity]).mean(0),dim=0) for identity in identities])
                features=torch.stack([f[feature] for f in qfeats])
                scores=features@centers.T
                ranks=scores.argsort(dim=1,descending=True,stable=True)
                positive=[]; negative=[]; fake=[]; correct1=correct5=0
                fake_correct1 = fake_correct5 = 0
                for i,q in enumerate(queries):
                    target=identities.index(str(q.get('target_id',q.get('identity'))))
                    hit1 = int(ranks[i, 0] == target)
                    hit5 = int((ranks[i, :min(5, len(identities))] == target).any())
                    correct1 += hit1
                    correct5 += hit5
                    if is_fake[i]:
                        fake_correct1 += hit1
                        fake_correct5 += hit5
                    positive.append(float(scores[i,target]))
                    negative.extend(scores[i,torch.arange(len(identities))!=target].tolist())
                    if 'source_id' in q:
                        source=identities.index(str(q['source_id']))
                        fake.append({'path':q['path'],'s_target':float(scores[i,target]),
                                     's_source':float(scores[i,source]),
                                     'target_minus_source':float(scores[i,target]-scores[i,source])})
                metrics={'queries':len(queries),'gallery_identities':len(identities),
                         'top1':correct1/len(queries),'top5':correct5/len(queries),
                         'same_identity_cosine_mean':sum(positive)/len(positive),
                         'other_identity_cosine_mean':sum(negative)/len(negative),
                         'clean_occluded_cosine_mean':float((features*torch.stack([f[feature] for f in clean])).sum(1).mean())}
                if fake:
                    metrics['fake_target_source_scores']=fake
                    metrics['target_gt_source_fraction']=sum(r['target_minus_source']>0 for r in fake)/len(fake)
                if is_fake.any():
                    count = int(is_fake.sum())
                    metrics['fake_top1'] = fake_correct1 / count
                    metrics['fake_top5'] = fake_correct5 / count
                    logging.info('[%s/%s] fake Accuracy: %.2f%%   Target Top-5: %.2f%%   '
                                 'Fake queries: %d   Gallery IDs: %d',
                                 mode, feature, 100 * metrics['fake_top1'], 100 * metrics['fake_top5'],
                                 count, len(identities))
                else:
                    logging.info('[%s/%s] Top-1: %.2f%%   Top-5: %.2f%%',
                                 mode, feature, 100 * metrics['top1'], 100 * metrics['top5'])
                result[mode][feature]=metrics
        report['checkpoints'][str(Path(checkpoint).resolve())]=result
        del model, fake_model
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    logging.info('Saved results: %s   Log: %s', out, log_path)

if __name__=='__main__':
    main()
