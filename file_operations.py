
import os
from pathlib import Path
import zipfile

class FileOperations:
    def analyze_directory(self, directory_path):
        """Analyze all files in a directory and return their details."""
        file_details = []
        total_size = 0
        for file_path_obj in Path(directory_path).glob('**/*'):
            if file_path_obj.is_file():
                size_kb = file_path_obj.stat().st_size / 1024
                extension = file_path_obj.suffix.lower()[1:] if file_path_obj.suffix else "No Extension"
                file_type = self.get_file_type(extension)
                file_details.append({
                    "path": str(file_path_obj),
                    "name": file_path_obj.name,
                    "extension": extension,
                    "type": file_type,
                    "size_kb": round(size_kb, 2)
                })
                total_size += size_kb
        return file_details, round(total_size, 2)

    def get_file_type(self, extension):
        """Classify file type based on extension."""
        image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'svg', 'webp']
        document_extensions = ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt', 'md']
        spreadsheet_extensions = ['xls', 'xlsx', 'csv', 'ods']
        presentation_extensions = ['ppt', 'pptx', 'key', 'odp']
        audio_extensions = ['mp3', 'wav', 'ogg', 'flac', 'aac', 'm4a']
        video_extensions = ['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm']
        code_extensions = ['py', 'java', 'cpp', 'c', 'h', 'js', 'html', 'css', 'php', 'rb', 'go']
        archive_extensions = ['zip', 'rar', '7z', 'tar', 'gz', 'bz2']

        if extension in image_extensions: return "Image"
        if extension in document_extensions: return "Document"
        if extension in spreadsheet_extensions: return "Spreadsheet"
        if extension in presentation_extensions: return "Presentation"
        if extension in audio_extensions: return "Audio"
        if extension in video_extensions: return "Video"
        if extension in code_extensions: return "Code"
        if extension in archive_extensions: return "Archive"
        return "Other"

    def compress_file(self, file_path):
        """Compress a file to zip format."""
        try:
            original_path = Path(file_path)
            zip_path = original_path.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(file_path, arcname=original_path.name)
            return True, str(zip_path)
        except Exception as e:
            return False, str(e)

    def create_new_file(self, directory, filename):
        """Create a new empty file in the specified directory."""
        try:
            file_path = Path(directory) / filename
            file_path.touch()
            return True, str(file_path)
        except Exception as e:
            return False, str(e)

    def delete_file(self, file_path):
        """Delete a file from the filesystem."""
        try:
            Path(file_path).unlink(missing_ok=True)
            return True, "File deleted successfully."
        except Exception as e:
            return False, str(e)