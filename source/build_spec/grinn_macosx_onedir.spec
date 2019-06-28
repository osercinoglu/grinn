# -*- mode: python -*-
import glob

block_cipher = None
# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('../resources/clover.ico','resources'),
('../VERSION','.'),
('/Users/onursercinoglu/Dropbox/gRINN_dist/samples','samples'),
('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/python2.7/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('../data','data'),
('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/python2.7/'
  'site-packages/panedr','panedr'),
('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/python2.7/'
  'site-packages/prody','prody')]

#python_bins = glob.glob('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/*.dylib')
#binaries = [(binary,binary.split('/')[-1]) for binary in python_bins]

binaries = [('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/libpython2.7.dylib','.'),
('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/libGLEW.1.13.dylib','.'),
('/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/libGLEW.1.13.0.dylib','.')]

a = Analysis(['../grinn.py'],
             pathex=['/Users/onursercinoglu/PycharmProjects/grinn/source',
                      '/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib',
                      '/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/python2.7',
                      '/Users/onursercinoglu/anaconda3/envs/grinn_py27/lib/python2.7/site-packages'],
             binaries=binaries,
             datas=datas,
             hiddenimports=['pymol.povray','pymol.opengl','pymol.parser','pymol','pymol.contrib',
             'pymol.bonds','pymol2','pandas._libs.tslibs.timedeltas','scipy._lib.messagestream'],
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
          debug=True,
          strip=False,
          upx=True,
          console=True,
          icon='resources/clover.ico')

coll = COLLECT(exe,
              a.binaries,
              a.zipfiles,
              a.datas,
              strip=False,
              upx=True,
              name='grinn')

# app = BUNDLE(exe,
#              name='grinn.app',
#              icon='resources/clover.icns',
#              bundle_identifier=None,
#              info_plist={
#               'NSHighResolutionCapable': 'True'
#              },)
