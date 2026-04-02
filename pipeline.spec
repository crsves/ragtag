# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# PIL/Pillow is imported by sentence_transformers.models.CLIPModel at module
# level; the native _imaging extension must be explicitly collected so
# multiprocessing worker processes can start up without crashing.
pil_datas, pil_binaries, pil_hiddenimports = collect_all('PIL')

a = Analysis(
    ['pipeline.py'],
    pathex=[],
    binaries=pil_binaries,
    datas=pil_datas,
    hiddenimports=pil_hiddenimports + ['tokenizers', 'tokenizers.tokenizers', 'PIL._imaging'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['rthooks/rthook_torch_inspect.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
