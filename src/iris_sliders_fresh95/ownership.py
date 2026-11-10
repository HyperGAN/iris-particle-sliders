"""Keep all old Iris guards, plus genuine ownership of this new namespace."""
import base64,hashlib,importlib.util
from pathlib import Path
from .util import source_record


def require():
    from iris_sliders.profiles import installation,verify_runtime
    dist,architecture=installation()
    document=verify_runtime()  # Unchanged full core/PG/profile/RECORD admission.
    root=Path(dist.locate_file('iris_sliders_fresh95')).absolute()
    if Path(__file__).absolute().parent!=root:
        raise RuntimeError('new Iris namespace must belong to the installed Iris distribution')
    expected={str(p):p for p in(dist.files or())if str(p).startswith('iris_sliders_fresh95/')and p.suffix=='.py'}
    actual={str(p.relative_to(root.parent)):p for p in root.rglob('*.py')if '__pycache__'not in p.parts}
    if set(expected)!=set(actual):raise RuntimeError('new namespace package/RECORD closure differs')
    for name,path in actual.items():
        pin=expected[name].hash
        actual_hash=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
        if pin is None or pin.mode!='sha256' or actual_hash!=pin.value:
            raise RuntimeError('new Iris namespace bytes differ from owning distribution RECORD')
    spec=importlib.util.find_spec('iris_sliders_fresh95')
    if spec is None or Path(spec.origin).absolute()!=root/'__init__.py':
        raise RuntimeError('new Iris namespace is shadowed')
    return architecture,document


def sources():
    from iris_sliders import adapters,train,native,teacher,profiles,cleanup
    from particle_sliders import pg09,pg09_images,pg09_backends
    own={p.name:source_record(p)for p in sorted(Path(__file__).absolute().parent.glob('*.py'))}
    base={name:source_record(module.__file__)for name,module in(
        ('adapter',adapters),('train',train),('native',native),('teacher',teacher),
        ('profiles',profiles),('cleanup',cleanup),('pg09',pg09),('pg09_images',pg09_images),('pg09_backends',pg09_backends))}
    return dict(own=own,original=base)
