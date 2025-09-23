import csv
from pathlib import Path
import argparse


def main(n=100):
    # repo root is two levels up from this script (scripts/ -> repo root)
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    src = data_dir / 'openpowerlifting.csv'
    sample = data_dir / 'openpowerlifting_sample.csv'

    if not src.exists():
        print(f'Source {src} not found. If you have the real dataset stored elsewhere, copy it to the repo root or run this script from the repo root.')
        return 1

    with src.open(newline='', encoding='utf-8') as src_f, sample.open('w', newline='', encoding='utf-8') as out_f:
        reader = csv.reader(src_f)
        writer = csv.writer(out_f)
        try:
            header = next(reader)
        except StopIteration:
            print('Source CSV appears empty')
            return 1
        writer.writerow(header)
        for i, row in enumerate(reader):
            writer.writerow(row)
            if i + 1 >= n:
                break

    print(f'Wrote sample {sample} with {n} rows.')
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a small sample of the openpowerlifting CSV in the repo root')
    parser.add_argument('-n', '--rows', type=int, default=100, help='number of data rows to include in the sample')
    args = parser.parse_args()
    raise SystemExit(main(args.rows))
