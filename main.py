import os
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess

from file_operations import FileOperations

class FileAnalyzerApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("File Analyzer Tool")
        self.root.geometry("900x700")

        self.file_ops = FileOperations()


        self.selected_directory = ""
        self.file_details = []
        self.current_sort_column = "Name"
        self.sort_direction_asc = True

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

        info_frame = ttk.Frame(self.root, padding="5")
        info_frame.pack(fill=tk.X)
        self.total_files_label = ttk.Label(info_frame, text="Total Files: 0")
        self.total_files_label.pack(side=tk.LEFT, padx=5)
        self.total_size_label = ttk.Label(info_frame, text="Total Size: 0 KB")
        self.total_size_label.pack(side=tk.LEFT, padx=5)

        tree_frame = ttk.Frame(self.root, padding="5")
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("Name", "Extension", "Type", "Size (KB)"), show="headings")

        columns_config = {
            "Name": {"width": 300, "stretch": tk.YES},
            "Extension": {"width": 100, "stretch": tk.NO},
            "Type": {"width": 120, "stretch": tk.NO},
            "Size (KB)": {"width": 100, "stretch": tk.NO, "anchor": tk.E}
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
    def browse_folder(self):
        directory = filedialog.askdirectory()
        if directory:
            self.folder_address_var.set(directory)
            self.analyze_folder()

    def analyze_folder(self):
        directory = self.folder_address_var.get()
        if not directory:
            messagebox.showerror("Error", "Please select or enter a folder path.")
            return
        if not os.path.isdir(directory):
            messagebox.showerror("Error", "Invalid directory path.")
            return

        self.selected_directory = directory
        for item in self.tree.get_children(): self.tree.delete(item)
        try:
            self.file_details, total_size = self.file_ops.analyze_directory(directory)
            self.total_files_label.config(text=f"Total Files: {len(self.file_details)}")
            self.total_size_label.config(text=f"Total Size: {total_size:.2f} KB")
            self.populate_main_treeview()
            self.sort_main_tree_data(self.current_sort_column, initial_sort=True)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during folder analysis: {str(e)}")

    def populate_main_treeview(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        for i, file_info in enumerate(self.file_details):
            self.tree.insert("", tk.END, iid=str(i), values=(
                file_info["name"], file_info["extension"],
                file_info["type"], f"{file_info['size_kb']:.2f}"
            ))

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
            "Size (KB)": lambda x: x["size_kb"]
        }
        if column_name in key_map:
            self.file_details.sort(key=key_map[column_name], reverse=not self.sort_direction_asc)
            self.populate_main_treeview()

        for col_id_str in self.tree["columns"]:
            self.tree.heading(col_id_str, text=col_id_str)
        header_text = f"{column_name} {'▲' if self.sort_direction_asc else '▼'}"
        self.tree.heading(column_name, text=header_text)

    def get_selected_file_info_from_main_tree(self):
        selected_items_iid = self.tree.selection()
        if not selected_items_iid:
            messagebox.showinfo("Info", "Please select a file from the list.")
            return None
        try:
            index = int(selected_items_iid[0])
            if 0 <= index < len(self.file_details):
                return self.file_details[index]
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