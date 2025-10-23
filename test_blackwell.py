#!/usr/bin/env python3
"""
Blackwell Architecture Support Test Script
Tests PyTorch installation and Blackwell (sm_120) GPU support
"""

import torch
import sys

def test_blackwell_support():
    """Test if PyTorch supports Blackwell architecture"""
    
    print("🔍 Testing Blackwell (sm_120) Architecture Support")
    print("=" * 60)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test Blackwell support")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # Check each GPU
    blackwell_gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        compute_capability = f"{props.major}.{props.minor}"
        
        print(f"GPU {i}: {props.name}")
        print(f"  - Compute Capability: {compute_capability}")
        print(f"  - Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  - Multiprocessors: {props.multi_processor_count}")
        
        # Blackwell architecture is sm_120 (compute capability 12.0)
        if props.major >= 12:
            print(f"  ✅ Blackwell architecture support detected!")
            blackwell_gpus.append(i)
        elif props.major == 11:
            print(f"  ⚠️  Hopper architecture (not Blackwell)")
        elif props.major == 10:
            print(f"  ⚠️  Ampere architecture (not Blackwell)")
        else:
            print(f"  ⚠️  Older architecture (not Blackwell)")
        print()
    
    # Test tensor operations on Blackwell GPUs
    if blackwell_gpus:
        print("🧪 Testing tensor operations on Blackwell GPUs...")
        try:
            for gpu_id in blackwell_gpus:
                print(f"Testing GPU {gpu_id}...")
                
                # Create a tensor on the GPU
                x = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
                y = torch.randn(1000, 1000, device=f'cuda:{gpu_id}')
                
                # Test basic operations
                z = torch.matmul(x, y)
                result = torch.sum(z)
                
                print(f"  ✅ Basic operations successful (result: {result.item():.2f})")
                
                # Test mixed precision if available
                if hasattr(torch.cuda, 'amp'):
                    with torch.cuda.amp.autocast():
                        z_fp16 = torch.matmul(x.half(), y.half())
                        print(f"  ✅ Mixed precision operations successful")
                
        except Exception as e:
            print(f"  ❌ Error during tensor operations: {e}")
            return False
    else:
        print("⚠️  No Blackwell GPUs detected")
        print("   Consider using the standard speedrun.sh script instead")
        return False
    
    print("🎉 Blackwell architecture support test completed successfully!")
    return True

def test_training_compatibility():
    """Test if the training scripts can run with Blackwell GPUs"""
    
    print("\n🚀 Testing Training Compatibility")
    print("=" * 60)
    
    try:
        # Test basic model creation
        import torch.nn as nn
        
        # Create a simple transformer-like model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(1000, 512)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(512, 8, batch_first=True),
                    num_layers=4
                )
                self.output = nn.Linear(512, 1000)
            
            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                x = self.output(x)
                return x
        
        model = SimpleModel().cuda()
        print("✅ Model creation successful")
        
        # Test forward pass
        batch_size, seq_len = 4, 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
        
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"✅ Forward pass successful (output shape: {output.shape})")
        
        # Test training step
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Forward pass with gradient computation
        output = model(input_ids)
        optimizer.zero_grad()
        loss = criterion(output.view(-1, 1000), input_ids.view(-1))
        loss.backward()
        optimizer.step()
        
        print(f"✅ Training step successful (loss: {loss.item():.4f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Training compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    print("NanoChat Blackwell Support Test")
    print("=" * 60)
    
    # Test Blackwell support
    blackwell_ok = test_blackwell_support()
    
    # Test training compatibility
    training_ok = test_training_compatibility()
    
    print("\n📋 Test Summary")
    print("=" * 60)
    print(f"Blackwell Support: {'✅ PASS' if blackwell_ok else '❌ FAIL'}")
    print(f"Training Compatibility: {'✅ PASS' if training_ok else '❌ FAIL'}")
    
    if blackwell_ok and training_ok:
        print("\n🎉 All tests passed! Ready for Blackwell training.")
        print("\n💡 Next steps:")
        print("   1. Run: bash install_blackwell.sh")
        print("   2. Run: bash speedrun.sh")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Please check your PyTorch installation.")
        print("\n💡 Troubleshooting:")
        print("   1. Make sure you're using PyTorch nightly build")
        print("   2. Verify CUDA 12.6+ is installed")
        print("   3. Check GPU compute capability (should be 12.0+)")
        sys.exit(1)
