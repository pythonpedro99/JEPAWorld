source .venv/bin/activate
deactivate
pip freeze > ../requirements.txt


 python Miniworld/scripts/manual_control.py --env-name JEPAENV-v0


(1) Miniworld/scripts/manual_control.py --env-name MiniWorld-PutNext-v0
(2) Miniworld/scripts/manual_control.py --env-name MiniWorld-FourRooms-v0
(3) Miniworld/scripts/manual_control.py --env-name MiniWorld-Hallway-v0
(3) Miniworld/scripts/manual_control.py --env-name MiniWorld-OneRoom-v0
(4) Miniworld/scripts/manual_control.py --env-name MiniWorld-PickupObjects-v0
(5) Miniworld/scripts/manual_control.py --env-name MiniWorld-ThreeRooms-v0
(6) Miniworld/scripts/manual_control.py --env-name MiniWorld-WallGap-v0  


export PYOPENGL_PLATFORM=osmesa
