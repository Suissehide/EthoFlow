import deeplabcut as dlc
CONFIG = "D:/LEO/dlc-projects/souris-bottomview-Leo-2026-06-05/config.yaml"
VIDEO = ["D:\\LEO\\dlc-projects\\souris-bottomview-Leo-2026-06-05\\result-videos\\970\\970DLC_HrnetW32_souris-bottomviewJun5shuffle1_snapshot_best-40_p60_labeled.mp4"]

dlc.create_labeled_video(CONFIG, VIDEO, pcutoff=0.1, draw_skeleton=True)