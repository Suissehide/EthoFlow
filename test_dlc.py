import deeplabcut

# Une de tes vraies vidéos
video = "/Users/couffinhal/Documents/Inserm/data_test/OF-test-30s.mp4"

deeplabcut.video_inference_superanimal(
    [video],
    superanimal_name="superanimal_topviewmouse",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    videotype="mp4",
    video_adapt=False,   # mettre True pour faire un fine-tune court — plus précis mais plus long
)