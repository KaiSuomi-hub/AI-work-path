##We need this to get the path to the files where we run the script
##Place this root of cats-dogs-kasperi
import os

"""
## Load the data: the Cats vs Dogs dataset
## First clean the data set
##This counter will keep track of the number of non-JFIF files that are found and removed.
##The script then iterates over three folders named "Cat", "Dog", and "Kasperi" inside a parent directory "PetImages".
For each folder, it constructs the full path to the folder using os.path.join("PetImages", folder_name).
"""

num_skipped = 0
for folder_name in ("Cat", "Dog", "Kasperi"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            """
            is_jfif = b"JFIF" in fobj.peek(10) is setting the variable is_jfif to
            True if the "JFIF" marker is found within the first 10 bytes of the file object fobj,
            indicating that the file is likely a JPEG image.
            """
            is_jfif = b"JFIF" in fobj.peek(10)
            bits = fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            ##Add a counter
            num_skipped += 1
            #report 10 bytes
            print(bits, " this is the 10 first bits")

print(f"Deleted {num_skipped} images.")