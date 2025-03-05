/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights
 *Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd.
 *and may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <unordered_map>

#include "AudioFile.h"
#include "EngineWrapper.hpp"
#include "Lexicon.hpp"
#include "OnnxWrapper.hpp"
#include "cmdline.hpp"

static std::vector<int> intersperse(const std::vector<int> &lst, int item) {
  std::vector<int> result(lst.size() * 2 + 1, item);
  for (size_t i = 1; i < result.size(); i += 2) {
    result[i] = lst[i / 2];
  }
  return result;
}

static int calc_product(const std::vector<int64_t> &dims) {
  int64_t result = 1;
  for (auto i : dims)
    result *= i;
  return result;
}

static double get_current_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);

  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char **argv) {
  cmdline::parser cmd;
  cmd.add<std::string>("encoder", 'e', "encoder onnx", false,
                       "../models/encoder-zh.onnx");
  cmd.add<std::string>("decoder", 'd', "decoder axmodel", false,
                       "../models/decoder-zh.axmodel");
  cmd.add<std::string>("lexicon", 'l', "lexicon.txt", false,
                       "../models/lexicon.txt");
  cmd.add<std::string>("token", 't', "tokens.txt", false,
                       "../models/tokens.txt");
  cmd.add<std::string>("g", 0, "g.bin", false, "../models/g-zh_mix_en.bin");

  cmd.add<std::string>(
      "sentence", 's', "input sentence", false,
      "爱芯元智半导体股份有限公司，致力于打造世界领先的人工智能感知与边缘计算芯"
      "片。服务智慧城市、智能驾驶、机器人的海量普惠的应用");
  cmd.add<std::string>("wav", 'w', "wav file", false, "output.wav");

  cmd.add<float>("speed", 0, "speak speed", false, 0.8f);
  cmd.add<int>("sample_rate", 0, "sample rate", false, 44100);
  cmd.parse_check(argc, argv);

  auto encoder_file = cmd.get<std::string>("encoder");
  auto decoder_file = cmd.get<std::string>("decoder");
  auto lexicon_file = cmd.get<std::string>("lexicon");
  auto token_file = cmd.get<std::string>("token");
  auto g_file = cmd.get<std::string>("g");

  auto sentence = cmd.get<std::string>("sentence");
  auto wav_file = cmd.get<std::string>("wav");

  auto speed = cmd.get<float>("speed");
  auto sample_rate = cmd.get<int>("sample_rate");

  printf("encoder: %s\n", encoder_file.c_str());
  printf("decoder: %s\n", decoder_file.c_str());
  printf("lexicon: %s\n", lexicon_file.c_str());
  printf("token: %s\n", token_file.c_str());
  printf("sentence: %s\n", sentence.c_str());
  printf("wav: %s\n", wav_file.c_str());
  printf("speed: %f\n", speed);
  printf("sample_rate: %d\n", sample_rate);

  int ret = AX_SYS_Init();
  if (0 != ret) {
    fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
    return -1;
  }

  AX_ENGINE_NPU_ATTR_T npu_attr;
  memset(&npu_attr, 0, sizeof(npu_attr));
  npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
  ret = AX_ENGINE_Init(&npu_attr);
  if (0 != ret) {
    fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
    return -1;
  }

  // Load lexicon
  Lexicon lexicon(lexicon_file, token_file);

  // Convert sentence to phones and tones
  std::vector<int> phones_bef, tones_bef;
  lexicon.convert(sentence, phones_bef, tones_bef);

  // for (auto p : phones_bef) {
  //     printf("%d ", p);
  // }
  // printf("\n");

  // Add blank between words
  auto phones = intersperse(phones_bef, 0);
  auto tones = intersperse(tones_bef, 0);
  int phone_len = phones.size();

  std::vector<int> langids(phone_len, 3);

  // Read g.bin
  std::vector<float> g(256, 0);
  FILE *fp = fopen(g_file.c_str(), "rb");
  if (!fp) {
    printf("Open %s failed!\n", g_file.c_str());
    return -1;
  }
  fread(g.data(), sizeof(float), g.size(), fp);
  fclose(fp);

  double start, end;

  start = get_current_time();
  OnnxWrapper encoder;
  if (0 != encoder.Init(encoder_file)) {
    printf("encoder init failed!\n");
    return -1;
  }
  end = get_current_time();
  printf("Load encoder take %.2f ms\n", (end - start));

  start = get_current_time();
  EngineWrapper decoder_model;
  if (0 != decoder_model.Init(decoder_file.c_str())) {
    printf("Init decoder model failed!\n");
    return -1;
  }
  end = get_current_time();
  printf("Load decoder take %.2f ms\n", (end - start));

  float noise_scale = 0.3f;
  float length_scale = 1.0 / speed;
  float noise_scale_w = 0.6f;
  float sdp_ratio = 0.2f;

  start = get_current_time();
  auto encoder_output = encoder.Run(phones, tones, langids, g, noise_scale,
                                    noise_scale_w, length_scale, sdp_ratio);
  float *zp_data = encoder_output.at(0).GetTensorMutableData<float>();
  int audio_len = encoder_output.at(2).GetTensorMutableData<int>()[0];
  auto zp_info = encoder_output.at(0).GetTensorTypeAndShapeInfo();
  auto zp_shape = zp_info.GetShape();
  end = get_current_time();
  printf("Encoder run take %.2f ms\n", (end - start));

  /*
printf("Load decoder model\n");
EngineWrapper decoder_model;
if (0 != decoder_model.Init(decoder_file.c_str())) {
  printf("Init decoder model failed!\n");
  return -1;
}
  */

  int zp_size = decoder_model.GetInputSize(0) / sizeof(float);
  int dec_len = zp_size / zp_shape[1];
  int audio_slice_len = decoder_model.GetOutputSize(0) / sizeof(float);
  std::vector<float> decoder_output(audio_slice_len);

  int dec_slice_num = int(std::ceil(zp_shape[2] * 1.0 / dec_len));
  std::vector<float> wavlist;
  start = get_current_time();
  for (int i = 0; i < dec_slice_num; i++) {
    std::vector<float> zp(zp_size, 0);
    int actual_size =
        (i + 1) * dec_len < zp_shape[2] ? dec_len : zp_shape[2] - i * dec_len;
    for (int n = 0; n < zp_shape[1]; n++) {
      memcpy(zp.data() + n * dec_len, zp_data + n * zp_shape[2] + i * dec_len,
             sizeof(float) * actual_size);
    }

    decoder_model.SetInput(zp.data(), 0);
    decoder_model.SetInput(g.data(), 1);
    if (0 != decoder_model.RunSync()) {
      printf("Run decoder model failed!\n");
      return -1;
    }
    decoder_model.GetOutput(decoder_output.data(), 0);

    actual_size = (i + 1) * audio_slice_len < audio_len
                      ? audio_slice_len
                      : audio_len - i * audio_slice_len;
    wavlist.insert(wavlist.end(), decoder_output.begin(),
                   decoder_output.begin() + actual_size);
  }
  end = get_current_time();
  printf("Decoder run %d times take %.2f ms\n", (end - start), dec_slice_num);
  printf("wav len: %d\n", wavlist.size());
  AudioFile<float> audio_file;
  std::vector<std::vector<float>> audio_samples{wavlist};
  audio_file.setAudioBuffer(audio_samples);
  audio_file.setSampleRate(sample_rate);
  if (!audio_file.save(wav_file)) {
    printf("Save audio file failed!\n");
    return -1;
  }

  printf("Saved audio to %s\n", wav_file.c_str());

  return 0;
}
