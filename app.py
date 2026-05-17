import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import librosa
import os
import tempfile

# ==========================================
# 1. 모델 구조 정의
# ==========================================
class FireworkCRNN(nn.Module):
    def __init__(self, hidden_size=32, num_layers=2):
        super(FireworkCRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=(1, 1)), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d((2, 1)), 
            nn.Conv2d(16, 32, (3, 3), padding=(1, 1)), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((4, 1))  
        )
        self.lstm = nn.LSTM(32 * 16 + 1, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        x_mel, x_onset = x[:, :, :128], x[:, :, 128:]
        cnn_out = self.cnn(x_mel.unsqueeze(1).transpose(2, 3))
        b, c, f, t = cnn_out.size()
        cnn_out = cnn_out.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        out, _ = self.lstm(torch.cat([cnn_out, x_onset], dim=-1))
        return self.fc(out).squeeze(-1)

# ==========================================
# 2. 오디오 전처리 함수
# ==========================================
def preprocess_audio(wav_path, fps=30):
    sr = 22050
    y, _ = librosa.load(wav_path, sr=sr)
    hop_length = int(sr / fps)
    
    mel_spec = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length), ref=np.max)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length).reshape(1, -1)
    
    X = np.vstack([mel_spec, onset_env]).T
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)
    return X_tensor

# ==========================================
# 3. AI 추론 및 큐 시퀀스 추출
# ==========================================
def generate_cues_dataframe(model, X_tensor, threshold=0.4, fps=30):
    with torch.no_grad():
        raw_predictions = model(X_tensor)
        probabilities = torch.sigmoid(raw_predictions).squeeze().numpy()
    
    cue_indices = np.where(probabilities >= threshold)[0]
    
    final_timestamps = []
    last_time = -1.0
    for idx in cue_indices:
        current_time = idx / fps
        if current_time - last_time >= 0.5: 
            final_timestamps.append(current_time)
            last_time = current_time
            
    if len(final_timestamps) > 0:
        df_output = pd.DataFrame({
            'Time (Seconds)': final_timestamps,
            'Effect Description': ['AI Beat Cue'] * len(final_timestamps)
        })
        return df_output
    return None

# ==========================================
# 4. Streamlit 웹 인터페이스 구성
# ==========================================
def main():
    st.set_page_config(page_title="Firework Cue AI", layout="centered")
    
    st.title("🎆 AI 불꽃놀이 큐(Cue) 자동 생성기")
    st.write("음악 파일(.wav)을 업로드하면, AI가 주파수와 비트를 분석하여 Finale 3D용 큐 스크립트(CSV)를 만들어줍니다.")

    model_path = "firework_ai_model.pth"
    
    if not os.path.exists(model_path):
        st.error(f"오류: 모델 파일('{model_path}')을 찾을 수 없습니다. app.py와 같은 폴더에 넣어주세요.")
        return

    @st.cache_resource
    def load_model():
        model = FireworkCRNN(hidden_size=32)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        return model

    model = load_model()

    st.sidebar.header("⚙️ 분석 설정")
    threshold = st.sidebar.slider("박자 감도 (Threshold)", min_value=0.01, max_value=1.0, value=0.4, step=0.05)
    fps = st.sidebar.number_input("목표 FPS", min_value=10, max_value=60, value=30, step=1)

    uploaded_file = st.file_uploader("여기에 WAV 오디오 파일을 드래그 앤 드롭 하세요.", type=["wav"])

    if uploaded_file is not None:
        st.info(f"선택된 파일: {uploaded_file.name}")
        
        if st.button("🚀 AI 분석 시작", use_container_width=True):
            with st.spinner("AI가 음악을 듣고 하이라이트를 분석하고 있습니다..."):
                try:
                    # 업로드된 파일을 임시 저장하여 librosa가 읽을 수 있게 처리
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    X_tensor = preprocess_audio(tmp_file_path, fps=fps)
                    df_result = generate_cues_dataframe(model, X_tensor, threshold=threshold, fps=fps)
                    
                    os.unlink(tmp_file_path) # 임시 파일 삭제
                    
                    if df_result is not None:
                        st.success(f"✨ 분석 성공! 총 {len(df_result)}개의 불꽃 큐가 생성되었습니다.")
                        
                        st.subheader("📋 큐 리스트 미리보기")
                        st.dataframe(df_result, use_container_width=True, height=200)
                        
                        csv_buffer = df_result.to_csv(index=False, encoding='utf-8-sig')
                        
                        st.download_button(
                            label="📥 생성된 CSV 파일 다운로드",
                            data=csv_buffer,
                            file_name=f"Cues_{os.path.splitext(uploaded_file.name)[0]}.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    else:
                        st.warning("⚠️ 감지된 박자가 없습니다. 왼쪽 사이드바에서 '박자 감도'를 낮추고 다시 시도해보세요.")
                        
                except Exception as e:
                    st.error(f"분석 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    main()