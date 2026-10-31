"""Check previous modern preview rejection without changing its packets or cohort."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from particle_sliders.inference import open_inference
from particle_sliders.lifecycle import SessionConfig, open_session
from iris_sliders.profiles import verify_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    original = json.loads((args.preview / 'preview-receipt.json').read_text())
    assert original['core_commit'] == '4abc0eb03a4e3a84fc3e6d02734c13dc7e8ea231'
    assert verify_runtime()['core_commit'] == 'a794e3b4a8813b012f2c4faa3da82b78d4a90214'
    for name, digest in original['files'].items():
        assert hashlib.sha256((args.preview / name).read_bytes()).hexdigest() == digest
    config = SessionConfig('iris', original['cohort'], **original['options'])
    reasons = {}
    with open_session(config) as session:
        before = session.checkpoint()['sha256']
        try:
            session.restore(torch.load(args.preview / 'training.pt', map_location='cpu', weights_only=True))
        except (ValueError, RuntimeError) as error:
            reasons['training.pt'] = str(error)
        else:
            raise AssertionError('the old training preview was silently accepted')
        assert session.checkpoint()['sha256'] == before
    for selection in ('raw', 'ema'):
        for kind in ('v1', 'native'):
            path = args.preview / (selection + '-' + kind + '.pt')
            rng = torch.get_rng_state().clone()
            try:
                if kind == 'native':
                    with open_inference(path, operation='response'):
                        raise AssertionError('the old native preview was silently accepted')
                else:
                    with open_session(config, artifact=path, operation='evaluate'):
                        raise AssertionError('the old export preview was silently accepted')
            except (ValueError, RuntimeError) as error:
                reasons[path.name] = str(error)
            assert torch.equal(rng, torch.get_rng_state())
    for name, digest in original['files'].items():
        assert hashlib.sha256((args.preview / name).read_bytes()).hexdigest() == digest
    args.receipt.write_text(json.dumps({'old_core_commit': original['core_commit'],
        'new_core_commit': verify_runtime()['core_commit'], 'old_files_unchanged': True,
        'training_restore_no_mutation': True, 'rejected_files': reasons,
        'native_and_export_refusal_preserves_rng': True}, indent=2) + '\n')
    print('All five previous preview packets were refused without relabeling or mutation.')


if __name__ == '__main__':
    main()
