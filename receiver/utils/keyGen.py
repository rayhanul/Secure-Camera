import base64
import os

enc = base64.b64encode(os.urandom(32)).decode()
hmac = base64.b64encode(os.urandom(32)).decode()

print("ENC_KEY_BASE64=", enc)
print("HMAC_KEY_BASE64=", hmac)
