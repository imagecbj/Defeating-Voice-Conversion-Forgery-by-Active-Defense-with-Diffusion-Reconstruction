import os
import torch.cuda


class HyParams:
    def __init__(self):
        self.project_root = ""
        self.sample_rate = 22050
        self.ref_db = 20.0
        self.dc_db = 100.0
        self.seed = 1240
        self.seg_len = 32
        self.seg_len_mel = 128    # mel spectrogram 
        self.hop_length = 256     # mel spectrogram  hop_length
        self.seg_len_audio = self.seg_len_mel * self.hop_length  #  audio 
        self.pm = 0.075
        self.epochs = 5
        self.bs = 4
        self.lr = 0.001
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.lambda_adv_l1 = 0
        self.lambda_adv_l2 = 0
        self.lambda_quality =1
        self.writer_info = f"pts/valid_test_adainvc"

        self.vctk_speaker_info = self.__path_join(
            self.project_root, "/home/lab/workspace/works/thy/data/vctk_speaker_info"
        )
        self.vctk_txt = self.__path_join(self.project_root, "data/VCTK-Corpus-0.92/txt")
        self.vctk_spk_txt = self.__path_join(self.project_root, "data/spk_txt.json")
        self.metadata = self.__path_join(self.project_root, "/home/lab/workspace/works/thy/data/metadata.json")
        self.gender_info = self.__path_join(self.project_root, "data/gender_info.json")
        self.vqvcp_model = self.__path_join(self.project_root, "target/vqvcp/gen")
        self.saved_pt = self.__path_join(self.project_root, "checkpoints/netG")
        self.temp_folder = self.__path_join(self.project_root, "temp")
        self.vctk_mels = self.__path_join(self.project_root, "")
        self.vctk_48k = self.__path_join(
            self.project_root, ""
        )
        self.vctk_22k = self.__path_join(self.project_root, ")
        self.vctk_rec_mels = self.__path_join(self.project_root, "data/rec_mels")
        self.vctk_eer = self.__path_join(
            self.project_root,
            "metrics/speaker_verification/equal_error_rate/VCTK_eer.yaml",
        )
        self.d_net = self.__path_join(self.project_root, "checkpoints/swcsm/d_net.pt")

        # adainvc
        self.adainvc_model = self.__path_join(self.project_root, "target/adainvc/model-995000.ckpt")
    
        # vqvcp
        self.vqvcp_model = self.__path_join(
            self.project_root, "target/vqvcp/gen")

        # againvc
        self.againvc_model = self.__path_join(
            self.project_root, 'target/againvc/c4s')

        # 
        self.saved_pt = self.__path_join(self.project_root,
                                         self.writer_info)

    def __path_join(self, project_root, relative_path):
        root = os.path.join(project_root, relative_path)
        return root
    # __init__  VCTK 


hp = HyParams()

if __name__ == "__main__":
    pass