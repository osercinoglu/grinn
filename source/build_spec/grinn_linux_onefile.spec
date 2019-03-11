# -*- mode: python -*-

block_cipher = None

# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('../resources/clover.ico','resources'),
('../VERSION','.'),
('.','.'),
('../samples','samples'),
('/home/onur/anaconda3/lib/python3.6/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('../data','data'),
('/home/onur/anaconda3/lib/python3.7/'
  'site-packages/panedr','panedr'),
('/home/onur/anaconda3/lib/python3.6/site-packages/PyQt5/',
'PyQt5/'),
('/home/onur/anaconda3/lib/python3.6/site-packages/prody/utilities/datafiles',
'prody/utilities/datafiles')]

binaries = [('../data/xcbglintegrations/libqxcb-glx-integration.so',
  'qt5_plugins/xcbglintegrations/'),('../data/xcbglintegrations/libqxcb-egl-integration.so','qt5_plugins/xcbglintegrations/'),
  ('/home/onur/anaconda3/lib/libiomp5.so','.')]

paths = ['/home/onur/repos/gRINN/source',
             '/home/onur/anaconda3/lib/python3.7/site-packages/',
             '/home/onur/anaconda3/lib/python3.6/site-packages/',
             '/home/onur/anaconda3/lib',
             '/home/onur/anaconda3/lib/python3.6/site-packages/PyQt5/']

a = Analysis(['../grinn.py'],
             pathex=paths,
             binaries=binaries,
             datas=datas,
             hiddenimports=['pymol.povray','pymol.parser',
             'pandas._libs.tslibs.timedeltas','pandas._libs.tslibs.np_datetime',
             'pandas._libs.tslibs.nattype','pandas._libs.skiplist','scipy._lib.messagestream','PyQt5.sip',
             'resultsGUI','resultsGUI_design','calcGUI','calcGUI_design','calc','corr','common','grinnGUI_design','pen'],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4','IPython'],
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