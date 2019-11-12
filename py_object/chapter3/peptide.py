class AudioFile:
    def __init__(self, filename):
        # 父类使用多肽来判别是否为合法子类
        if not filename.split('.')[1] in self.ext:
            raise Exception("Invalid format")

        self.filename = filename


class Mp3File(AudioFile):
    ext = "mp3"

    def play(self):
        print("playing mp3")


class WavFile(AudioFile):
    ext = "wav"

    def play(self):
        print("playing wav")


class OggFile(AudioFile):
    ext = "ogg"

    def play(self):
        print("playing ogg")


if __name__ == '__main__':
    ogg = OggFile("hello.ogg")
    ogg.play()
