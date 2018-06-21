# -*- mode: python -*-
import sys
sys.setrecursionlimit(5000)
block_cipher = None

options = [('v', None, 'OPTION')]

# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('../resources/clover.ico','resources'),
('../VERSION','.'),
('../samples','samples'),
('/home/onur/anaconda2/lib/python2.7/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('../data','data'),
('/home/onur/anaconda2/lib/python2.7/'
  'site-packages/panedr','panedr')]

binaries = [('../data/xcbglintegrations/libqxcb-glx-integration.so',
  'qt5_plugins/xcbglintegrations/'),('../data/xcbglintegrations/libqxcb-egl-integration.so',
  'qt5_plugins/xcbglintegrations/')]

paths = ['/home/onur/repos/gRINN/source',
             '/home/onur/anaconda2/lib/python2.7/site-packages/',
             '/home/onur/anaconda2/lib']

a = Analysis(['../grinn.py'],
             pathex=paths,
             binaries=binaries,
             datas=datas,
             hiddenimports=['pymol.povray','pymol.parser',
             'pandas._libs.tslibs.timedeltas','scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4','PySide'],
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
          options,
          exclude_binaries=False,
          name='grinn',
          debug=True,
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