# -*- mode: python -*-
import glob

block_cipher = None
# Icons and other stuff that I discover to be necessary by trial-and-error
datas = [('../resources/clover.ico','resources'),
('../VERSION','.'),
('/Users/onur/anaconda2/lib/python2.7/'
  'site-packages/mdtraj/formats/pdb/data',
  'mdtraj/formats/pdb/data'),
('../data','data'),
('/Users/onur/anaconda2/lib/python2.7/'
  'site-packages/panedr','panedr')]

#python_bins = glob.glob('/Users/onur/anaconda2/lib/*.dylib')
#binaries = [(binary,binary.split('/')[-1]) for binary in python_bins]

binaries = [('/Users/onur/anaconda2/lib/libpython2.7.dylib','.')]

a = Analysis(['../grinn.py'],
             pathex=['/Users/onur/repos/gRINN/source',
                      '/Users/onur/anaconda2/lib',
                      '/Users/onur/anaconda2/lib/python2.7',
                      '/Users/onur/anaconda2/lib/python2.7/site-packages'],
             binaries=binaries,
             datas=datas,
             hiddenimports=['pymol.povray','pymol.parser',
                            'pandas._libs.tslibs.timedeltas','scipy._lib.messagestream'],
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
          name='grinn',
          debug=False,
          strip=False,
          upx=True,
          console=False,
          icon='resources/clover.ico')

#coll = COLLECT(exe,
#               a.binaries,
#               a.zipfiles,
#               a.datas,
#               strip=False,
#               upx=True,
#               name='grinn')

app = BUNDLE(exe,
             name='grinn.app',
             icon='resources/clover.icns',
             bundle_identifier=None,
             info_plist={
              'NSHighResolutionCapable': 'True'
             },)
