# -*- mode: python -*-

block_cipher = None

# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('clover.ico','.'),
('/home/onur/anaconda2/lib/python2.7/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('/home/onur/anaconda2/etc/fonts','etc/fonts'),
('/usr/share/X11/xkb','usr/share/X11/xkb'),
('/home/onur/anaconda2/lib/python2.7/'
  'site-packages/panedr','panedr')]

binaries = [('/home/onur/anaconda2/plugins/xcbglintegrations/libqxcb-glx-integration.so',
  'qt5_plugins/xcbglintegrations/')]

paths = ['/home/onur/repos/gRINN/getresinten',
             '/home/onur/anaconda2/lib/python2.7/site-packages/',
             '/home/onur/anaconda2/lib']

a = Analysis(['grinn.py'],
             pathex=paths,
             binaries=binaries,
             datas=datas,
             hiddenimports=['pymol.povray','pymol.parser',
             'pandas._libs.tslibs.timedeltas','scipy._lib.messagestream'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='grinn',
          debug=True,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='grinn')
