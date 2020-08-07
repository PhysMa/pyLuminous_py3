#!/usr/bin/env python
"""installs the sopra spectral data"""
import urllib.request, urllib.error, urllib.parse
import zipfile
import datetime,time
import os,platform
import shutil

url =  "http://www.sspectra.com/files/misc/win/SOPRA.EXE"
outdir = 'sopra'

## Download sopra data archive (it's an executable zipfile)
f = urllib.request.urlopen(url)
with open("SOPRA.EXE","wb") as download:
    download.write(f.read())

if platform.system() == "Windows":
    ## Extracting files from archive
    #zipfile.is_zipfile("SOPRA.EXE")
    z = zipfile.ZipFile("SOPRA.EXE")
    z.extractall(outdir) #extract to directory sopra
    
    ## Setting creation times
    try:
        import win32file
        
        def setWinCreationTime(fileName,creationTime):
            """creationTime - datetime object"""
            filehandle = win32file.CreateFile(fileName,win32file.GENERIC_WRITE, 0, None, win32file.OPEN_EXISTING, 0, None)
            theCreationTime = theLastAccessTime = theLastWriteTime = creationTime
            win32file.SetFileTime(filehandle, theCreationTime, theLastAccessTime, theLastWriteTime)
        
        for info in z.infolist():
            fname = os.path.join(outdir,info.filename)
            creationTime = datetime.datetime(*info.date_time)
            setWinCreationTime(fname,creationTime)
    except:
        print("unable to alter creation times of files", file=sys.stderr)

elif platform.system() == "Linux": #there is no creation time in linux and can use os.utime().
    ## Extracting files from archive    
    def fixzip(zipfile):
        zipFileContainer = open(zipfile,'r+b')
        # HACK: See http://bugs.python.org/issue10694
        # The zip file generated is correct, but because of extra data after the 'central directory' section,
        # Some version of python (and some zip applications) can't read the file. By removing the extra data,
        # we ensure that all applications can read the zip without issue.
        # The ZIP format: http://www.pkware.com/documents/APPNOTE/APPNOTE-6.3.0.TXT
        # Finding the end of the central directory:
        #   http://stackoverflow.com/questions/8593904/how-to-find-the-position-of-central-directory-in-a-zip-file
        #   http://stackoverflow.com/questions/20276105/why-cant-python-execute-a-zip-archive-passed-via-stdin
        #       This second link is only losely related, but echos the first, "processing a ZIP archive often requires backwards seeking"
        content = zipFileContainer.read()
        pos = content.rfind('\x50\x4b\x05\x06') # reverse find: this string of bytes is the end of the zip's central directory.
        if pos>0:
            zipFileContainer.seek(pos+20) # +20: see secion V.I in 'ZIP format' link above.
            zipFileContainer.truncate()
            zipFileContainer.write('\x00\x00') # Zip file comment length: 0 byte length; tell zip applications to stop reading.
            zipFileContainer.seek(0)
        return zipFileContainer
    
    shutil.copy("SOPRA.EXE","SOPRA.ZIP")
    z_fh = fixzip("SOPRA.ZIP") # this fix is only necessary on older versions of python. python 2.7+ wouldn't need it.
    
    z = zipfile.ZipFile(z_fh)
    z.extractall(outdir)
    
    ## Setting creation times
    for info in z.infolist():
        fname = os.path.join(outdir,info.filename)
        creationTime = datetime.datetime(*info.date_time)
        mtime = atime = time.mktime(creationTime.timetuple())
        os.utime(fname,(atime,mtime))
    

## Getting info from zipfile (not necessary)

def print_info(archive_name):
    zf = zipfile.ZipFile(archive_name)
    for info in zf.infolist():
        print(info.filename)
        print('\tComment:\t', info.comment)
        print('\tModified:\t', datetime.datetime(*info.date_time))
        print('\tSystem:\t\t', info.create_system, '(0 = Windows, 3 = Unix)')
        print('\tZIP version:\t', info.create_version)
        print('\tCompressed:\t', info.compress_size, 'bytes')
        print('\tUncompressed:\t', info.file_size, 'bytes')
        print()

