# -*- mode: python -*-

block_cipher = None

binaries = [('/home/onur/anaconda2/lib/python2.7/site-packages/vmd/libvmd.so',
'vmd'),
('/home/onur/anaconda2/lib/libtcl8.5.so',
'.')]

a = Analysis(['../makedrypsf.py'],
             pathex=['/home/onur/repos/gRINN/source'],
             binaries=binaries,
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='makedrypsf',
          debug=False,
          strip=False,
          upx=True,
          console=True )
