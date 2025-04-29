source .venv/bin/activate
deactivate
pip freeze > ../requirements.txt



scripts/manual_control.py --env-name MiniWorld-RoomObjects-v0 