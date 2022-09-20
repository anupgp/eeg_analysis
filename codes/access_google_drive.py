from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

credentials_file_path = "/Users/macbookair/goofy/.ssh/google_credentials.json"
clientsecret_file_path = "/Users/macbookair/goofy/.ssh/google_client_secret.json"

gauth = GoogleAuth()
# Try to load saved client credentials
gauth.LoadCredentialsFile(credentials_file_path)
if gauth.credentials is None:
    # Authenticate if they're not there
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    # Refresh them if expired
    gauth.Refresh()
else:
    # Initialize the saved creds
    gauth.Authorize()
# Save the current credentials to a file
 # gauth.SaveCredentialsFile(credentials_file_path)

drive = GoogleDrive(gauth)

def ListFolder(parent):
  filelist=[]
  file_list = drive.ListFile({'q': "'%s' in parents and trashed=false" % parent}).GetList()
  for f in file_list:
    if f['mimeType']=='application/vnd.google-apps.folder': # if folder
        filelist.append({"id":f['id'],"title":f['title'],"list":ListFolder(f['id'])})
        print('folder title: %s, id: %s' % (f['title'], f['id']))
    else:
        filelist.append({"title":f['title'],"title1":f['alternateLink']})
        print('file title: %s, id: %s' % (f['title'], f['id']))
  return filelist


# ListFolder('root')
ListFolder('1xohXvvIAwmVq7rqh9-BKIt_Ye_aTJkCo')
# Auto-iterate through all files in the root folder.
 # file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
# folderid = "i1VPT6DfZ09yFjq5dzUIA0_Wecm1vsc3"
# file_list = drive.ListFile({'q': "'1i1VPT6DfZ09yFjq5dzUIA0_Wecm1vsc3' in parents and trashed=false"}).GetList()
# for file1 in file_list:
#   print('title: %s, id: %s' % (file1['title'], file1['id']))


  
# fpath = 'Computers/My MacBook Air/Anup_2TB/raw_data/astron/raw/astrocyte/frap30scarel_processed/';
# fname = "frap30scacytrel_crosscorr.mat"
# Initialize GoogleDriveFile instance with file id
# gfile = drive.CreateFile()
# Read the file and set it as its content
# gfile.SetContentFile(fpath+fname)
# textfile.SetContentFile('/Users/macbookair/texput.log')
# textfile.Upload()
# print(textfile)
# drive.CreateFile({'id':textfile['id']}).GetContentFile('eng-dl.txt')
