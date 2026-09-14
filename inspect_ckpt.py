import torch
ck = torch.load(r".\ckpt\panocontext_v1 (1).pth", map_location="cpu")
print("top-level type:", type(ck))
if isinstance(ck, dict):
    print("top-level keys:", list(ck.keys())[:10])
sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
ks = list(sd.keys())
print("num tensors:", len(ks))
print("first 15 keys:")
for k in ks[:15]:
    print("  ", k)