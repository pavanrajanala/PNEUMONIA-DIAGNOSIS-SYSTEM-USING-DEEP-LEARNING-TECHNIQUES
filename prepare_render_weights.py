"""Download and verify the published research weights during a Render build."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NAME = 'nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt'
URL = 'https://github.com/mlmed/torchxrayvision/releases/download/v1/' + NAME
SHA256 = '56524913dd16a906422e8d8b66a7a5c46be1d82eb7ac012d8103776f1aa68899'
DESTINATION = ROOT / 'weights' / NAME


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    DESTINATION.parent.mkdir(exist_ok=True)
    if DESTINATION.exists() and digest(DESTINATION) == SHA256:
        print('Verified pretrained weights already present.')
        return

    temporary = DESTINATION.with_suffix('.download')
    try:
        print('Downloading pretrained research weights for Render...', flush=True)
        with urlopen(URL, timeout=120) as response, temporary.open('wb') as target:
            while block := response.read(1024 * 1024):
                target.write(block)
        if digest(temporary) != SHA256:
            raise RuntimeError('Downloaded weights did not match the expected SHA-256.')
        temporary.replace(DESTINATION)
        print('Weights verified.', flush=True)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
