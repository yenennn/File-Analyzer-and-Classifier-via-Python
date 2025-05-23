import os
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess

from file_operations import FileOperations
from text_classifier import TextClassifier

try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    print("PyMuPDF (fitz) not found. PDF text extraction will be unavailable. Install with 'pip install PyMuPDF'")

try:
    import docx
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False
    print("python-docx not found. DOCX text extraction will be unavailable. Install with 'pip install python-docx'")


TF_AVAILABLE = False
IMAGE_MODEL_AVAILABLE = False

ResNet50 = None
preprocess_input = None
decode_predictions = None
image = None
np = None

try:
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import ResNet50 as ResNet50_keras
    from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_keras
    from tensorflow.keras.applications.resnet50 import decode_predictions as decode_predictions_keras
    from tensorflow.keras.preprocessing import image as image_keras
    import numpy as numpy_keras

    ResNet50 = ResNet50_keras
    preprocess_input = preprocess_input_keras
    decode_predictions = decode_predictions_keras
    image = image_keras
    np = numpy_keras
    TF_AVAILABLE = True
    print("TensorFlow and Keras components loaded successfully.")
except ImportError:
    print("TensorFlow or its Keras components not found. Image classification feature will be disabled.")
    print("To enable, please ensure TensorFlow is installed correctly (e.g., 'pip install tensorflow').")

def extract_text_from_file(file_path):
    """
    Extracts text content from a given file.
    Handles common text file extensions, PDF, and DOCX.
    Attempts to read with common encodings for text files.

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text content, or an empty string if extraction fails
             or the file type is not supported for text extraction.
    """
    text_extensions = {
        '.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv',
        '.log', '.ini', '.yaml', '.yml', '.rtf', '.tex', '.c', '.cpp', '.h',
        '.java', '.php', '.rb', '.sh', '.bat', '.ps1', '.sql'
    }
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.pdf':
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(file_path)
                text = ""
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                doc.close()
                return text
            else:
                print(f"PyMuPDF not available, cannot extract text from PDF: {file_path}")
                return ""
        elif file_ext == '.docx':
            if PYTHON_DOCX_AVAILABLE:
                doc = docx.Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            else:
                print(f"python-docx not available, cannot extract text from DOCX: {file_path}")
                return ""
        elif file_ext == '.doc':
            print(f".doc file ({file_path}) text extraction is not supported by this basic function.")
            return ""
        elif file_ext in text_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                        return f.read()
                except Exception as e_latin1:
                    print(f"Could not read {file_path} with UTF-8 or Latin-1: {e_latin1}")
                    return ""
            except Exception as e_utf8:
                print(f"Could not read {file_path} with UTF-8: {e_utf8}")
                return ""
        else:
            return ""
    except IOError as e_io:
        print(f"IOError reading {file_path}: {e_io}")
        return ""
    except Exception as e_general:
        print(f"General error processing {file_path} for text extraction: {e_general}")
        return ""


class FileAnalyzerApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("File Analyzer Tool")
        self.root.geometry("900x700")

        self.file_ops = FileOperations()
        self.text_classifier = TextClassifier()
        self.model_loaded_successfully = False
        try:
            self.text_classifier.load_model("text_classifier_model.joblib")
            print("Custom text classifier loaded successfully.")
            self.model_loaded_successfully = True
        except Exception as e:
            print(
                f"Could not load custom text classifier: {e}. It may need to be trained first using text_classifier.py.")
        self.image_classifier = None
        _image_model_loaded_init_time = False
        if TF_AVAILABLE and ResNet50:
            try:
                print("Initializing ResNet50 model for image classification...")
                self.image_classifier = ResNet50(weights='imagenet')
                _image_model_loaded_init_time = True
                print("ResNet50 model initialized successfully.")
            except Exception as e:
                print(f"Error initializing ResNet50 model: {e}")
                self.image_classifier = None
                _image_model_loaded_init_time = False
        else:
            if not TF_AVAILABLE:
                print("TensorFlow is not available, ResNet50 model not initialized.")
            elif not ResNet50:
                print("ResNet50 component not loaded (import issue), model not initialized.")
            _image_model_loaded_init_time = False

        globals()['IMAGE_MODEL_AVAILABLE'] = _image_model_loaded_init_time


        self.current_sort_column = "Name"
        self.sort_direction_asc = True
        self.selected_directory = ""
        self.file_details = []

        self.filterable_columns = ["Name", "Extension", "Type", "Subject"]
        self.filter_values = {col: tk.StringVar() for col in self.filterable_columns}
        self.filter_key_map = {
            "Name": "name",
            "Extension": "extension",
            "Type": "type",
            "Subject": "subject"
        }
        self.create_widgets()


    def create_widgets(self):
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Folder Path:").pack(side=tk.LEFT, padx=(0, 5))
        self.folder_address_var = tk.StringVar()
        self.folder_address_entry = ttk.Entry(top_frame, textvariable=self.folder_address_var, width=60)
        self.folder_address_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.browse_button = ttk.Button(top_frame, text="Browse...", command=self.browse_folder)
        self.browse_button.pack(side=tk.LEFT, padx=(5, 0))

        buttons_frame = ttk.Frame(self.root, padding="5")
        buttons_frame.pack(fill=tk.X)
        self.analyze_button = ttk.Button(buttons_frame, text="Analyze Folder", command=self.analyze_folder)
        self.analyze_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.create_file_button = ttk.Button(buttons_frame, text="Create New File",
                                             command=self.create_new_file_in_current_dir)
        self.create_file_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.compress_button = ttk.Button(buttons_frame, text="Compress File",
                                          command=self.compress_selected_file)
        self.compress_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.delete_button = ttk.Button(buttons_frame, text="Delete File",
                                        command=self.delete_selected_file)
        self.delete_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.ml_classify_button = ttk.Button(buttons_frame, text="Classify Content (ML)",
                                             command=self.enhanced_content_classification)
        self.ml_classify_button.pack(side=tk.LEFT, padx=5, pady=5)

        info_frame = ttk.Frame(self.root, padding="5")
        info_frame.pack(fill=tk.X)
        self.total_files_label = ttk.Label(info_frame, text="Total Files: 0")
        self.total_files_label.pack(side=tk.LEFT, padx=5)
        self.total_size_label = ttk.Label(info_frame, text="Total Size: 0 KB")
        self.total_size_label.pack(side=tk.LEFT, padx=5)

        filter_bar_frame = ttk.Frame(self.root, padding="5")
        filter_bar_frame.pack(fill=tk.X)

        for col_name in self.filterable_columns:
            ttk.Label(filter_bar_frame, text=f"{col_name}:").pack(side=tk.LEFT, padx=(5, 2))
            filter_entry = ttk.Entry(filter_bar_frame, textvariable=self.filter_values[col_name], width=12)
            filter_entry.pack(side=tk.LEFT, padx=(0, 5))
            filter_entry.bind("<Return>", lambda event: self.apply_filters())

        self.apply_filters_button = ttk.Button(filter_bar_frame, text="Apply Filters", command=self.apply_filters)
        self.apply_filters_button.pack(side=tk.LEFT, padx=5)
        self.clear_filters_button = ttk.Button(filter_bar_frame, text="Clear Filters", command=self.clear_filters)
        self.clear_filters_button.pack(side=tk.LEFT, padx=5)

        tree_frame = ttk.Frame(self.root, padding="5")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("Name", "Extension", "Type", "Size (KB)", "Subject"),
                                 show="headings")

        columns_config = {
            "Name": {"width": 250, "stretch": tk.YES},
            "Extension": {"width": 80, "stretch": tk.NO},
            "Type": {"width": 100, "stretch": tk.NO},
            "Size (KB)": {"width": 100, "stretch": tk.NO, "anchor": tk.E},
            "Subject": {"width": 150, "stretch": tk.NO}  # Configuration for the new column
        }
        for col, config in columns_config.items():
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_main_tree_data(c))
            self.tree.column(col, width=config["width"], stretch=config["stretch"], anchor=config.get("anchor", tk.W))

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<Button-3>", self.show_main_tree_context_menu)
        self.tree.bind("<Double-1>", self.open_selected_file_from_event)

    def apply_filters(self):
        """Applies current filters and refreshes the treeview."""
        self.populate_main_treeview()

    def clear_filters(self):
        """Clears all filter entries and refreshes the treeview."""
        for col_var in self.filter_values.values():
            col_var.set("")
        self.populate_main_treeview()

    def browse_folder(self):
        directory = filedialog.askdirectory()
        if directory:
            self.folder_address_var.set(directory)
            self.analyze_folder()

    def analyze_folder(self):
        directory = self.folder_address_var.get()
        if not directory or not os.path.isdir(directory):
            messagebox.showerror("Error", "Please select or enter a valid folder path.")
            return
        self.selected_directory = directory
        self.clear_filters()
        try:
            self.file_details, total_size = self.file_ops.analyze_directory(directory)
            self.total_files_label.config(text=f"Total Files: {len(self.file_details)}")
            self.total_size_label.config(text=f"Total Size: {total_size:.2f} KB")
            self.populate_main_treeview()
            self.sort_main_tree_data(self.current_sort_column, initial_sort=True)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during folder analysis: {str(e)}")

    def populate_main_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        active_filters = {}
        for col_name, str_var in self.filter_values.items():
            filter_text = str_var.get().lower().strip()
            if filter_text:
                active_filters[col_name] = filter_text

        displayed_count = 0
        for i, file_info in enumerate(self.file_details):
            match = True
            if active_filters:
                for col_name, filter_text in active_filters.items():
                    internal_key = self.filter_key_map.get(col_name)
                    if internal_key:
                        value_to_check = str(file_info.get(internal_key, "")).lower()
                        if filter_text not in value_to_check:
                            match = False
                            break

            if match:
                subject = file_info.get("subject", "N/A")
                self.tree.insert("", tk.END, iid=str(i), values=(
                    file_info["name"],
                    file_info["extension"],
                    file_info["type"],
                    f"{file_info['size_kb']:.2f}",
                    subject
                ))
                displayed_count += 1

        if active_filters:
            self.total_files_label.config(
                text=f"Total Files: {displayed_count} (filtered from {len(self.file_details)})")
        else:
            self.total_files_label.config(text=f"Total Files: {len(self.file_details)}")

    def sort_main_tree_data(self, column_name, initial_sort=False):
        if not self.file_details: return

        if not initial_sort and self.current_sort_column == column_name:
            self.sort_direction_asc = not self.sort_direction_asc
        else:
            self.sort_direction_asc = True
        self.current_sort_column = column_name

        key_map = {
            "Name": lambda x: x["name"].lower(),
            "Extension": lambda x: x["extension"].lower(),
            "Type": lambda x: x["type"].lower(),
            "Size (KB)": lambda x: x["size_kb"],
            "Subject": lambda x: x.get("subject", "").lower()
        }
        if column_name in key_map:
            self.file_details.sort(key=key_map[column_name], reverse=not self.sort_direction_asc)
            self.populate_main_treeview()

        for col_id_str in self.tree["columns"]:
            current_text = self.tree.heading(col_id_str, "text").replace(' ▲', '').replace(' ▼', '')
            self.tree.heading(col_id_str, text=current_text)
        header_text = f"{column_name} {'▲' if self.sort_direction_asc else '▼'}"
        self.tree.heading(column_name, text=header_text)

    def get_selected_file_info_from_main_tree(self):
        selected_items_iid = self.tree.selection()
        if not selected_items_iid:
            messagebox.showinfo("Info", "Please select a file from the list.")
            return None
        try:
            original_index = int(selected_items_iid[0])
            if 0 <= original_index < len(self.file_details):
                return self.file_details[original_index]
        except (ValueError, IndexError):
            pass
        messagebox.showerror("Error", "Could not map selection to file data.")
        return None

    def compress_selected_file(self):
        file_info = self.get_selected_file_info_from_main_tree()
        if not file_info: return
        success, result = self.file_ops.compress_file(file_info["path"])
        if success:
            messagebox.showinfo("Success", f"File compressed: {result}")
            self.analyze_folder()
        else:
            messagebox.showerror("Error", f"Compression failed: {result}")

    def create_new_file_in_current_dir(self):
        if not self.selected_directory:
            messagebox.showinfo("Info", "Please analyze a folder first to set a directory.")
            return
        filename_with_path = filedialog.asksaveasfilename(
            initialdir=self.selected_directory, title="Create New File", defaultextension=".txt",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename_with_path:
            dir_path = os.path.dirname(filename_with_path)
            base_name = os.path.basename(filename_with_path)
            success, result = self.file_ops.create_new_file(dir_path, base_name)
            if success:
                messagebox.showinfo("Success", f"File created: {result}")
                self.analyze_folder()
            else:
                messagebox.showerror("Error", f"Creation failed: {result}")

    def delete_selected_file(self):
        file_info = self.get_selected_file_info_from_main_tree()
        if not file_info: return
        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{file_info['name']}'?"):
            success, result = self.file_ops.delete_file(file_info["path"])
            if success:
                messagebox.showinfo("Success", result)
                self.analyze_folder()
            else:
                messagebox.showerror("Error", f"Deletion failed: {result}")

    def open_selected_file(self):
        file_info = self.get_selected_file_info_from_main_tree()
        if not file_info: return
        self._open_file_os(file_info["path"])

    def open_selected_file_from_event(self, event):
        item_iid = self.tree.identify_row(event.y)
        if item_iid:
            self.open_selected_file()

    def open_selected_file_location(self):
        file_info = self.get_selected_file_info_from_main_tree()
        if not file_info: return
        self._open_location_os(file_info["path"])

    def copy_selected_file_path(self):
        file_info = self.get_selected_file_info_from_main_tree()
        if not file_info: return
        self._copy_path_to_clipboard(file_info["path"])

    def show_main_tree_context_menu(self, event):
        item_iid = self.tree.identify_row(event.y)
        if not item_iid: return
        self.tree.selection_set(item_iid)

        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="Open", command=self.open_selected_file)
        menu.add_command(label="Open Location", command=self.open_selected_file_location)
        menu.add_separator()
        menu.add_command(label="Copy Path", command=self.copy_selected_file_path)
        menu.add_separator()
        menu.add_command(label="Compress", command=self.compress_selected_file)
        menu.add_command(label="Delete", command=self.delete_selected_file)
        menu.post(event.x_root, event.y_root)


    def _open_file_os(self, file_path):
        try:
            if os.name == 'nt':
                os.startfile(file_path)
            elif os.name == 'posix':
                opener = "xdg-open" if Path("/usr/bin/xdg-open").exists() else "open"
                subprocess.call([opener, file_path])
            else:
                webbrowser.open(Path(file_path).as_uri())
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file '{Path(file_path).name}': {e}")

    def _open_location_os(self, file_path):
        folder_path = os.path.dirname(file_path)
        try:
            if os.name == 'nt':
                subprocess.run(['explorer', folder_path], check=False)
            elif os.name == 'posix':
                opener = "xdg-open" if Path("/usr/bin/xdg-open").exists() else "open"
                subprocess.call([opener, folder_path])
            else:
                webbrowser.open(Path(folder_path).as_uri())
        except Exception as e:
            messagebox.showerror("Error", f"Could not open location for '{Path(file_path).name}': {e}")

    def _copy_path_to_clipboard(self, file_path):
        self.root.clipboard_clear()
        self.root.clipboard_append(file_path)
        messagebox.showinfo("Copied", "File path copied to clipboard.")

    def _display_classification_results(self):
        """Updates the treeview to show classification results."""
        self.populate_main_treeview()
        # Optionally, re-sort or inform user
        messagebox.showinfo("Classification Complete", "File subjects have been updated based on content analysis.")

    def enhanced_content_classification(self):
        if not self.file_details:
            messagebox.showinfo("Info", "Please analyze a folder first!")
            return

        missing_libs_messages = []
        if not TF_AVAILABLE or not IMAGE_MODEL_AVAILABLE:
            missing_libs_messages.append("TensorFlow and/or ResNet50 model (for image analysis)")

        if missing_libs_messages:
            messagebox.showwarning(
                "Missing Components",
                f"ML classification may be limited. Missing: {', '.join(missing_libs_messages)}."
            )

        progress_window = tk.Toplevel(self.root)
        progress_window.title("ML Content Analysis")
        progress_window.geometry("400x150")
        progress_label = tk.Label(progress_window, text="Initializing ML analysis...\nThis may take a while...")
        progress_label.pack(pady=10)
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=350, mode="determinate",
                                       variable=progress_var)
        progress_bar.pack(pady=10)
        status_label = tk.Label(progress_window, text="Preparing...")
        status_label.pack(pady=5)

        def update_progress(current, total, message):
            if total > 0:
                progress_var.set((current / total) * 100)
            status_label.config(text=message)
            progress_window.update()

        try:
            for file_item in self.file_details:
                if "subject" not in file_item:
                    file_item["subject"] = "Other"

            total_files = len(self.file_details)
            document_contents = []
            document_indices = []
            image_files_paths = []
            image_indices = []

            update_progress(0, total_files, "Scanning files for content type...")
            for i, file_item in enumerate(self.file_details):
                file_path = file_item["path"]
                if file_item["type"] == "Image" and \
                        TF_AVAILABLE and \
                        IMAGE_MODEL_AVAILABLE and \
                        self.image_classifier and \
                        image and np and preprocess_input and decode_predictions:
                    image_files_paths.append(file_path)
                    image_indices.append(i)
                elif file_item["type"] in ["Text", "Document", "Code"]:
                    content = extract_text_from_file(file_path)
                    if content and len(content.split()) > 5:
                        document_contents.append(content)
                        document_indices.append(i)
                    else:
                        file_item["subject"] = file_item.get("type", "Misc") + " File (Short/Empty)"
                else:
                    file_item["subject"] = file_item.get("type", "Misc") + " File"

            if self.model_loaded_successfully and self.text_classifier and document_contents:
                update_progress(0, len(document_contents), "Classifying text documents...")
                _, predicted_subjects_named = self.text_classifier.predict(document_contents)
                for i, original_file_index in enumerate(document_indices):
                    self.file_details[original_file_index]["subject"] = predicted_subjects_named[i]
                status_label.config(text="Text classification complete.")
                progress_window.update()
            elif document_contents:
                status_label.config(text="Text model not loaded or no text documents for custom classification.")
                progress_window.update()

            if self.image_classifier and image_files_paths:
                print(f"Attempting to classify {len(image_files_paths)} images...")  # Debug
                update_progress(0, len(image_files_paths), "Classifying image files...")
                for i, original_file_index in enumerate(image_indices):
                    file_path = image_files_paths[i]
                    try:
                        img = image.load_img(file_path, target_size=(224, 224))
                        img_array = image.img_to_array(img)
                        img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                        processed_img = preprocess_input(img_array_expanded_dims)

                        predictions = self.image_classifier.predict(processed_img)
                        decoded_preds = decode_predictions(predictions, top=1)[0]

                        if decoded_preds:
                            subject_description = decoded_preds[0][1]
                            self.file_details[original_file_index][
                                "subject"] = f"Image: {subject_description.replace('_', ' ').title()}"
                        else:
                            self.file_details[original_file_index]["subject"] = "Image (Unclassified)"
                    except Exception as e:
                        print(f"Error classifying image '{file_path}': {e}")
                        self.file_details[original_file_index]["subject"] = "Image (Classification Error)"
                    update_progress(i + 1, len(image_files_paths), f"Classified image {i + 1}/{len(image_files_paths)}")
                status_label.config(text="Image classification complete.")
                progress_window.update()
            elif image_files_paths:
                status_label.config(text="Image classifier/model not available. Skipping image classification.")
                for original_file_index in image_indices:
                    self.file_details[original_file_index]["subject"] = "Image (Unclassified - Model N/A)"
                progress_window.update()
            else:
                print("No images found or image classifier not ready.")

            progress_label.config(text="Classification complete!")
            progress_bar.config(value=100)
            status_label.config(text="Done.")
            progress_window.update()
            progress_window.after(1500, progress_window.destroy)

            self._display_classification_results()

        except Exception as e:
            if progress_window.winfo_exists():
                progress_window.destroy()
            messagebox.showerror("Error", f"An error occurred during ML content analysis: {str(e)}")
            import traceback
            traceback.print_exc()
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        elif 'vista' in available_themes:
            style.theme_use('vista')
    except tk.TclError:
        print("ttk themes not fully available or custom themes may not be supported on this system.")

    app = FileAnalyzerApp(root)
    root.mainloop()