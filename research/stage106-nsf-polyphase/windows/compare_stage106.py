import argparse
import json
from pathlib import Path


def compare(folder):
    path = Path(folder)
    records = [json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted(path.glob('*.benchmark.json'))]
    cases = {}
    for record in records:
        if record.get('stage') != 106 or not record.get('full_output_parity'):
            raise ValueError(f"unverified benchmark record: {record}")
        key = (record.get('operator','conv_transpose1d'), record['case'])
        cases.setdefault(key,{})[record['mode']] = record
    results=[]
    for (operator,case),modes in sorted(cases.items()):
        candidate_name = 'tiled' if operator == 'single_input_conv1d' else 'polyphase'
        if 'direct' not in modes or candidate_name not in modes:
            raise ValueError(f'missing direct or {candidate_name}: {operator}/{case}')
        direct=modes['direct']
        candidate=modes[candidate_name]
        a=float(direct['median_ms'])
        b=float(candidate['median_ms'])
        results.append({'operator':operator,'case':case,'direct_ms':a,'candidate_ms':b,
                        'speedup':a/b,'candidate_max_abs':candidate['max_abs'],
                        'candidate_snr_db':candidate.get('snr_db')})
    print(json.dumps({'stage':106,'comparisons':results},ensure_ascii=False,indent=2))
    (path/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    return results


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder')
    args=p.parse_args()
    compare(args.folder)
