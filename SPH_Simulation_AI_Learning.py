# GAN 학습 코드 (입자 시뮬레이션 csv 기반) - 메모리 최적화 버전
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import gc

# 1. 메모리 효율적인 슬라이딩 윈도우 Dataset 클래스 정의
class SlidingWindowDataset(Dataset):
    def __init__(self, csv_file, window_size=5, step_size=1, mode='train', train_ratio=0.7, val_ratio=0.15, max_particles=500):
        self.csv_file = csv_file
        self.window_size = window_size
        self.step_size = step_size
        self.mode = mode
        self.max_particles = max_particles
        
        # CSV 파일 정보만 먼저 확인
        print("CSV 파일 정보 확인 중...")
        sample_data = pd.read_csv(csv_file, nrows=1000)
        print("CSV 파일 컬럼:", sample_data.columns.tolist())
        
        # 전체 프레임 수 확인 (메모리 효율적으로)
        frame_counts = pd.read_csv(csv_file, usecols=['frame'])
        total_frames = frame_counts['frame'].nunique()
        print(f"총 {total_frames}개 프레임 발견")
        
        # 슬라이딩 윈도우 시퀀스 생성
        self.sequences = list(range(0, total_frames - window_size + 1, step_size))
        
        # 데이터 분할 (훈련/검증/테스트)
        total_sequences = len(self.sequences)
        train_end = int(total_sequences * train_ratio)
        val_end = int(total_sequences * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.sequences = self.sequences[:train_end]
        elif mode == 'val':
            self.sequences = self.sequences[train_end:val_end]
        elif mode == 'test':
            self.sequences = self.sequences[val_end:]
        
        print(f"{mode} 데이터: {len(self.sequences)}개 시퀀스")
        
        # 메모리 정리
        del sample_data, frame_counts
        gc.collect()

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start_idx = self.sequences[idx]
        
        # 필요한 프레임만 로드
        window_frames = []
        for i in range(self.window_size):
            frame_idx = start_idx + i
            
            # 해당 프레임의 데이터만 로드
            frame_data = pd.read_csv(self.csv_file, 
                                   dtype={'frame': 'int32', 'x': 'float32', 'y': 'float32', 
                                         'vx': 'float32', 'vy': 'float32'},
                                   usecols=['frame', 'x', 'y', 'vx', 'vy'])
            frame_data = frame_data[frame_data['frame'] == frame_idx]
            
            # 입자 수 제한
            if len(frame_data) > self.max_particles:
                frame_data = frame_data.sample(n=self.max_particles, random_state=42)
            
            # 필요한 컬럼만 선택하고 numpy 배열로 변환
            frame_values = frame_data[["x", "y", "vx", "vy"]].values.astype('float32')
            
            # 패딩 (고정 크기 보장)
            if len(frame_values) < self.max_particles:
                padding = np.zeros((self.max_particles - len(frame_values), 4), dtype='float32')
                frame_values = np.vstack([frame_values, padding])
            
            window_frames.append(frame_values)
        
        # GAN용 데이터 형태: (sequence_length, num_particles, features)
        sequence_data = np.array(window_frames)  # (window_size, max_particles, 4)
        
        # 입력: 마지막 프레임을 제외한 모든 프레임, 타겟: 마지막 프레임
        input_sequence = sequence_data[:-1]  # (window_size-1, max_particles, 4)
        target_frame = sequence_data[-1]     # (max_particles, 4)
        
        return torch.tensor(input_sequence, dtype=torch.float32), torch.tensor(target_frame, dtype=torch.float32)

# 2. Generator 모델 정의
class ParticleGenerator(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_particles=500, noise_dim=100):
        super(ParticleGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        self.noise_dim = noise_dim
        
        # 입력 처리 레이어
        self.input_processor = nn.Sequential(
            nn.Linear(input_size * (num_particles), hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 잡음 처리 레이어
        self.noise_processor = nn.Sequential(
            nn.Linear(noise_dim, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 결합된 특징 처리
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, input_size * num_particles),
            nn.Tanh()  # 출력을 [-1, 1] 범위로 정규화
        )
        
    def forward(self, input_sequence, noise):
        batch_size = input_sequence.shape[0]
        
        # 입력 시퀀스 처리 (마지막 프레임만 사용)
        last_frame = input_sequence[:, -1, :, :]  # (batch_size, num_particles, features)
        flattened_input = last_frame.view(batch_size, -1)  # (batch_size, num_particles * features)
        
        # 입력 특징 추출
        input_features = self.input_processor(flattened_input)
        
        # 잡음 처리
        noise_features = self.noise_processor(noise)
        
        # 특징 결합
        combined_features = torch.cat([input_features, noise_features], dim=1)
        
        # 결합된 특징 처리
        processed_features = self.combined_processor(combined_features)
        
        # 출력 생성
        output = self.output_layer(processed_features)
        
        # 출력 형태 재구성
        output = output.view(batch_size, self.num_particles, self.input_size)
        
        return output

# 3. Discriminator 모델 정의
class ParticleDiscriminator(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_particles=500):
        super(ParticleDiscriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_particles = num_particles
        
        # 입력 처리 레이어
        self.input_processor = nn.Sequential(
            nn.Linear(input_size * num_particles, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 특징 추출 레이어
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # 출력 레이어 (진짜/가짜 판별)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, particle_frame):
        batch_size = particle_frame.shape[0]
        
        # 입력 평탄화
        flattened_input = particle_frame.view(batch_size, -1)
        
        # 입력 처리
        processed_input = self.input_processor(flattened_input)
        
        # 특징 추출
        features = self.feature_extractor(processed_input)
        
        # 진짜/가짜 판별
        validity = self.output_layer(features)
        
        return validity

# 4. 데이터셋 및 모델 초기화
csv_path = r"C:\Users\sunma\particles_simulation_data.csv"

# CSV 파일 존재 확인
if not os.path.exists(csv_path):
    print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
    print("먼저 기본 시뮬레이션을 실행하여 CSV 파일을 생성해주세요.")
    exit()

# 메모리 효율적인 데이터셋 생성
print("데이터셋 생성 중...")
train_dataset = SlidingWindowDataset(csv_path, window_size=5, step_size=1, mode='train', max_particles=500)
val_dataset = SlidingWindowDataset(csv_path, window_size=5, step_size=1, mode='val', max_particles=500)
test_dataset = SlidingWindowDataset(csv_path, window_size=5, step_size=1, mode='test', max_particles=500)

# 작은 배치 크기로 메모리 사용량 제한
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

# GAN 모델 초기화
generator = ParticleGenerator(input_size=4, hidden_size=256, num_particles=500, noise_dim=100)
discriminator = ParticleDiscriminator(input_size=4, hidden_size=256, num_particles=500)

# 손실 함수와 옵티마이저
adversarial_loss = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

print("=== GAN 학습 시작 (메모리 최적화 버전) ===")
print(f"훈련 데이터: {len(train_dataset)}개 시퀀스")
print(f"검증 데이터: {len(val_dataset)}개 시퀀스")
print(f"테스트 데이터: {len(test_dataset)}개 시퀀스")
print(f"슬라이딩 윈도우 크기: 5")
print(f"배치 크기: 4")
print(f"최대 입자 수: 500")
print(f"Generator hidden_size: 256")
print(f"Discriminator hidden_size: 256")
print(f"잡음 차원: 100")
print(f"학습 에포크: 1000")
print("=====================")

# 학습 및 검증 함수 (메모리 최적화)
def train_epoch(generator, discriminator, dataloader, generator_optimizer, discriminator_optimizer, adversarial_loss, device):
    generator.train()
    discriminator.train()
    
    total_g_loss = 0
    total_d_loss = 0
    batch_count = 0
    
    for input_sequence, target_frame in dataloader:
        batch_size, seq_len, num_particles, features = input_sequence.shape
        
        # 메모리 절약을 위해 배치 크기 제한
        if batch_size * num_particles > 4000:
            continue
            
        # 데이터를 디바이스로 이동
        input_sequence = input_sequence.to(device)
        target_frame = target_frame.to(device)
        
        # 진짜/가짜 라벨 생성
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # ---------------------
        #  Discriminator 학습
        # ---------------------
        discriminator_optimizer.zero_grad()
        
        # 진짜 데이터에 대한 판별
        real_validity = discriminator(target_frame)
        d_real_loss = adversarial_loss(real_validity, valid)
        
        # 가짜 데이터 생성 및 판별
        noise = torch.randn(batch_size, 100).to(device)
        fake_frame = generator(input_sequence, noise)
        fake_validity = discriminator(fake_frame.detach())
        d_fake_loss = adversarial_loss(fake_validity, fake)
        
        # 전체 Discriminator 손실
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        d_loss.backward()
        discriminator_optimizer.step()
        
        # -----------------
        #  Generator 학습
        # -----------------
        generator_optimizer.zero_grad()
        
        # 가짜 데이터에 대한 판별 (Generator 관점)
        fake_validity = discriminator(fake_frame)
        g_loss = adversarial_loss(fake_validity, valid)
        
        g_loss.backward()
        generator_optimizer.step()
        
        total_g_loss += g_loss.item()
        total_d_loss += d_loss.item()
        batch_count += 1
        
        # 메모리 정리
        del input_sequence, target_frame, fake_frame, noise
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return total_g_loss / batch_count if batch_count > 0 else 0, total_d_loss / batch_count if batch_count > 0 else 0

def validate_epoch(generator, discriminator, dataloader, adversarial_loss, device):
    generator.eval()
    discriminator.eval()
    
    total_g_loss = 0
    total_d_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for input_sequence, target_frame in dataloader:
            batch_size, seq_len, num_particles, features = input_sequence.shape
            
            if batch_size * num_particles > 4000:
                continue
                
            # 데이터를 디바이스로 이동
            input_sequence = input_sequence.to(device)
            target_frame = target_frame.to(device)
            
            # 진짜/가짜 라벨 생성
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)
            
            # 가짜 데이터 생성
            noise = torch.randn(batch_size, 100).to(device)
            fake_frame = generator(input_sequence, noise)
            
            # 판별 결과
            real_validity = discriminator(target_frame)
            fake_validity = discriminator(fake_frame)
            
            # 손실 계산
            d_real_loss = adversarial_loss(real_validity, valid)
            d_fake_loss = adversarial_loss(fake_validity, fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            g_loss = adversarial_loss(fake_validity, valid)
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            batch_count += 1
            
            # 메모리 정리
            del input_sequence, target_frame, fake_frame, noise
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return total_g_loss / batch_count if batch_count > 0 else 0, total_d_loss / batch_count if batch_count > 0 else 0

# 평가 함수 (RMSE, MAE, R² 등) - 메모리 최적화
def evaluate_model(generator, test_loader, device):
    generator.eval()
    
    total_rmse = 0
    total_mae = 0
    total_r2 = 0
    batch_count = 0
    
    with torch.no_grad():
        for input_sequence, target_frame in test_loader:
            batch_size, seq_len, num_particles, features = input_sequence.shape
            
            if batch_size * num_particles > 4000:
                continue
                
            # 데이터를 디바이스로 이동
            input_sequence = input_sequence.to(device)
            target_frame = target_frame.to(device)
            
            # 예측 (Generator 사용)
            noise = torch.randn(batch_size, 100).to(device)
            predicted_frame = generator(input_sequence, noise)
            
            # RMSE 계산
            mse = torch.mean((target_frame - predicted_frame) ** 2)
            rmse = torch.sqrt(mse)
            
            # MAE 계산
            mae = torch.mean(torch.abs(target_frame - predicted_frame))
            
            # R² 계산 (간단한 버전)
            ss_res = torch.sum((target_frame - predicted_frame) ** 2)
            ss_tot = torch.sum((target_frame - torch.mean(target_frame)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            total_rmse += rmse.item()
            total_mae += mae.item()
            total_r2 += r2.item()
            batch_count += 1
            
            # 메모리 정리
            del input_sequence, target_frame, predicted_frame, noise
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return (total_rmse / batch_count if batch_count > 0 else 0,
            total_mae / batch_count if batch_count > 0 else 0,
            total_r2 / batch_count if batch_count > 0 else 0)

# 메인 학습 루프 (RMSE 기반) - 메모리 최적화
best_rmse = float('inf')
target_rmse = 0.04
patience = 30
patience_counter = 0
max_epochs = 2000

print(f"목표 RMSE: {target_rmse}")
print(f"최대 에포크: {max_epochs}")
print("="*50)

for epoch in range(max_epochs):
    # 훈련
    train_g_loss, train_d_loss = train_epoch(generator, discriminator, train_loader, 
                                            generator_optimizer, discriminator_optimizer, 
                                            adversarial_loss, device)
    
    # 검증
    val_g_loss, val_d_loss = validate_epoch(generator, discriminator, val_loader, 
                                           adversarial_loss, device)
    
    # RMSE 계산 (매 에포크마다)
    current_rmse, current_mae, current_r2 = evaluate_model(generator, test_loader, device)
    
    # 최고 성능 모델 저장 (RMSE 기준)
    if current_rmse < best_rmse:
        best_rmse = current_rmse
        patience_counter = 0
        # 최고 성능 모델 저장
        torch.save(generator.state_dict(), "best_gan_generator.pth")
        torch.save(discriminator.state_dict(), "best_gan_discriminator.pth")
        print(f"새로운 최고 RMSE: {best_rmse:.6f} (Epoch {epoch})")
    else:
        patience_counter += 1
    
    # 진행 상황 출력 (매 20 에포크마다)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:4d} | G Loss: {train_g_loss:.6f} | D Loss: {train_d_loss:.6f}")
        print(f"Val G Loss: {val_g_loss:.6f} | Val D Loss: {val_d_loss:.6f}")
        print(f"현재 RMSE: {current_rmse:.6f} | 목표: {target_rmse:.6f} | Best RMSE: {best_rmse:.6f}")
        print("-" * 50)
    
    # 목표 RMSE 달성 체크
    if current_rmse <= target_rmse:
        print(f"목표 RMSE 달성! Epoch {epoch}에서 RMSE: {current_rmse:.6f}")
        break
    
    # Early stopping 체크 (목표 달성 실패 시)
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (RMSE: {current_rmse:.6f})")
        break
    
    # 메모리 정리
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 최고 성능 모델 로드
generator.load_state_dict(torch.load("best_gan_generator.pth"))
discriminator.load_state_dict(torch.load("best_gan_discriminator.pth"))

# 테스트 데이터로 최종 평가
print("\n=== 최종 모델 평가 ===")
rmse, mae, r2 = evaluate_model(generator, test_loader, device)
print(f"최종 RMSE: {rmse:.6f}")
print(f"최종 MAE: {mae:.6f}")
print(f"최종 R² Score: {r2:.6f}")

# 목표 달성 여부 확인
if rmse <= target_rmse:
    print(f"✅ 목표 RMSE {target_rmse} 달성 성공!")
else:
    print(f"❌ 목표 RMSE {target_rmse} 달성 실패 (현재: {rmse:.6f})")

# 모델 저장
generator_save_path = "final_gan_generator.pth"
discriminator_save_path = "final_gan_discriminator.pth"
torch.save(generator.state_dict(), generator_save_path)
torch.save(discriminator.state_dict(), discriminator_save_path)
print(f"\n=== GAN 학습 완료 (메모리 최적화 버전) ===")
print(f"Generator 모델이 {generator_save_path}에 저장되었습니다.")
print(f"Discriminator 모델이 {discriminator_save_path}에 저장되었습니다.")
print("이제 보간 시뮬레이션에서 GAN 모델을 사용할 수 있습니다.")
