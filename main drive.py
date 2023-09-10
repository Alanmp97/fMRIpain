from onedrive_data import *

file_to_feed = FilesNamesAndTypes(mri_type=2, num_subject=(124,77))
file_to_feed.select_files_name_type()


losarchivos = file_to_feed.get_files()
