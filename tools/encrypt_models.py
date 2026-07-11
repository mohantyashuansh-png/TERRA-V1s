import os
import secrets
from pathlib import Path
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

def encrypt_file(file_path, key):
    with open(file_path, 'rb') as f:
        data = f.read()
    
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(data) + padder.finalize()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    out_path = Path("vault") / (Path(file_path).stem + ".terravis")
    with open(out_path, 'wb') as f:
        f.write(iv + encrypted_data)
    print(f"Encrypted {file_path} -> {out_path}")

def main():
    key = b'\xeb\xa06[\xb3\x91]\xc7\x02\x0eG\xff\x0c\x01\x91\x86\x8d\x9a\x90o\xc0\xe7\xcf\xf4\xc80\x15\xf9\x07c\x0c\xe6'
    print(f"Using existing AES_KEY.")
    
    Path("vault").mkdir(exist_ok=True)
    
    files = [
        "outputs/checkpoints/best_model.pth",
        "outputs/desert_seg.onnx",
        "sentinel_model.pth",
        "yolov8n.pt",
        "osnet_x0_25_msmt17.pt"
    ]
    
    for file in files:
        if os.path.exists(file):
            encrypt_file(file, key)
        else:
            print(f"Warning: {file} not found")

if __name__ == '__main__':
    main()
