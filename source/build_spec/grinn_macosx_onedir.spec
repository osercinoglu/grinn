# -*- mode: python -*-

block_cipher = None
# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('clover.ico','.'),
('/Users/onur/anaconda2/lib/python2.7/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('/Users/onur/anaconda2/lib/python2.7/'
  'site-packages/panedr','panedr')]

a = Analysis(['grinn.py'],
             pathex=['/Users/onur/repos/gRINN/getresinten'],
             binaries=[],
             datas=datas,
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
          exclude_binaries=True,
          name='grinn',
          debug=False,
          strip=False,
          upx=True,
          console=True )

#coll = COLLECT(exe,
#               a.binaries,
#               a.zipfiles,
#               a.datas,
#               strip=False,
#               upx=True,
#               name='grinn')
