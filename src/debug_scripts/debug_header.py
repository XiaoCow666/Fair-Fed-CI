import os

def debug():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "raw_data.csv")
    
    print(f"Reading {raw_path}...")
    with open(raw_path, 'rb') as f:
        header = f.readline()
        print(f"Raw header bytes: {header}")
        try:
            print(f"Decoded (gbk): {header.decode('gbk')}")
        except:
            print("Decoded (gbk): Failed")
            
        try:
            print(f"Decoded (utf-8): {header.decode('utf-8')}")
        except:
            print("Decoded (utf-8): Failed")

if __name__ == "__main__":
    debug()
