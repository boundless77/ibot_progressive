"""Fixed Stage-0 teacher/head diagnostics on identical masked patch positions.

The common Stage-0 head is an evaluation readout, never loaded into a trained model.
It provides a fixed prototype coordinate system, but penalizes encoder basis drift;
it is a diagnostic, not proof of semantic or identity recovery.
"""
import _bootstrap
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from progressive.evaluation import load_full_role, fixed_input

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True,help='Fixed Stage-0 full checkpoint')
    p.add_argument('--checkpoints',nargs='+',required=True)
    p.add_argument('--manifest',required=True,help='One relative real-image path per line')
    p.add_argument('--data_root',required=True)
    p.add_argument('--parsing_root',required=True)
    p.add_argument('--mask_bank',required=True)
    p.add_argument('--mode',choices=['single','double','triple','inner'],default='triple')
    p.add_argument('--mask_ratio', type=float, help='Fill a fixed mask budget for single/double/triple')
    p.add_argument('--device',default='cuda')
    p.add_argument('--temperature',type=float,default=.07)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.temperature<=0:
        raise ValueError('Temperature must be positive')
    if args.mask_ratio is not None and (not 0 < args.mask_ratio < 1 or args.mode == 'inner'):
        p.error('mask_ratio must be in (0,1) and requires single, double or triple mode')
    paths=[line.strip() for line in Path(args.manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not paths:
        raise ValueError('Empty evaluation manifest')
    reference,center=load_full_role(args.reference,args.device,'teacher')
    if center is None:
        raise ValueError('Reference must contain iBOT center2 for the fixed prototype readout')
    center=center.to(args.device)
    results={'reference':str(Path(args.reference).resolve()),'mode':args.mode,'mask_ratio':args.mask_ratio,
             'note':'Common Stage-0 head/teacher; sensitive to feature-basis drift. Not an identity metric.',
             'checkpoints':{}}
    for checkpoint in args.checkpoints:
        student,_=load_full_role(checkpoint,args.device,'student')
        rows=[]
        for path in paths:
            full_path=Path(args.data_root)/path
            image,_=fixed_input(full_path,args.data_root,None,'clean',None)
            _,mask=fixed_input(full_path,args.data_root,args.parsing_root,args.mode,args.mask_bank,
                               mask_ratio=args.mask_ratio)
            image=image.unsqueeze(0).to(args.device)
            mask=torch.from_numpy(mask).unsqueeze(0).to(args.device)
            clean_tokens=reference.backbone(image)
            masked_tokens = (student.masked_tokens(image, mask) if hasattr(student, 'decoder')
                             else student.backbone(image, mask=mask))
            reference_logits=reference.head(clean_tokens)[1]
            common_logits=reference.head(masked_tokens)[1]
            target=F.softmax((reference_logits-center)/args.temperature,-1)
            prediction=F.log_softmax((common_logits-center)/args.temperature,-1)
            kl=(target*(target.clamp_min(1e-12).log()-prediction)).sum(-1)[mask.flatten(1)]
            cosine=F.cosine_similarity(masked_tokens[:,1:],clean_tokens[:,1:],dim=-1)[mask.flatten(1)]
            top1=(reference_logits.argmax(-1)==common_logits.argmax(-1))[mask.flatten(1)]
            rows.append({'path':path,'masked_patches':int(mask.sum()),
                         'fixed_reference_kl':float(kl.mean()),
                         'fixed_reference_patch_cosine':float(cosine.mean()),
                         'fixed_reference_prototype_agreement':float(top1.float().mean())})
        results['checkpoints'][str(Path(checkpoint).resolve())]={'per_image':rows,
              'image_mean':{k:sum(r[k] for r in rows)/len(rows) for k in
                 ('fixed_reference_kl','fixed_reference_patch_cosine','fixed_reference_prototype_agreement')}}
        del student
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(results,indent=2),encoding='utf-8'); print(out)

if __name__=='__main__':
    main()
