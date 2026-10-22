"""扫码登录微信：uv run --with 'qrcode[pil]' python -m kairos.wechat"""

from ..config import get_app_dir
from .api import login

login(get_app_dir() / "channels" / "weixin" / "account.json")
