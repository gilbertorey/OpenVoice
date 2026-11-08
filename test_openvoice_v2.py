#!/usr/bin/env python3
"""
OpenVoice V2 Installation Test Script
This script tests the basic functionality of OpenVoice V2 installation.
"""

import os
import torch
import sys

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        from melo.api import TTS
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_checkpoints():
    """Test if checkpoints are available."""
    print("Testing checkpoints...")
    
    # Check V2 checkpoints
    converter_path = 'checkpoints_v2/checkpoints_v2/converter'
    base_speakers_path = 'checkpoints_v2/checkpoints_v2/base_speakers'
    
    if not os.path.exists(converter_path):
        print(f"❌ Converter checkpoints not found at: {converter_path}")
        return False
    
    if not os.path.exists(base_speakers_path):
        print(f"❌ Base speakers not found at: {base_speakers_path}")
        return False
    
    # Check for specific files
    config_file = os.path.join(converter_path, 'config.json')
    checkpoint_file = os.path.join(converter_path, 'checkpoint.pth')
    
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    if not os.path.exists(checkpoint_file):
        print(f"❌ Checkpoint file not found: {checkpoint_file}")
        return False
    
    print("✅ All checkpoints found!")
    return True

def test_device_setup():
    """Test device setup and model loading."""
    print("Testing device setup...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        from openvoice.api import ToneColorConverter
        
        ckpt_converter = 'checkpoints_v2/checkpoints_v2/converter'
        tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
        tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
        
        print("✅ ToneColorConverter loaded successfully!")
        return True, tone_color_converter
    except Exception as e:
        print(f"❌ Error loading ToneColorConverter: {e}")
        return False, None

def test_melo_tts():
    """Test MeloTTS functionality."""
    print("Testing MeloTTS...")
    
    try:
        from melo.api import TTS
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Test with English
        model = TTS(language='EN_NEWEST', device=device)
        print("✅ MeloTTS model loaded successfully!")
        
        # Check available speakers
        speaker_ids = model.hps.data.spk2id
        print(f"Available speakers: {list(speaker_ids.keys())}")
        
        return True
    except Exception as e:
        print(f"❌ Error with MeloTTS: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 OpenVoice V2 Installation Test")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    
    # Test 2: Checkpoints
    if test_checkpoints():
        tests_passed += 1
    
    # Test 3: Device setup and model loading
    success, converter = test_device_setup()
    if success:
        tests_passed += 1
    
    # Test 4: MeloTTS
    if test_melo_tts():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 OpenVoice V2 installation is working correctly!")
        print("\nNext steps:")
        print("1. Place your reference audio file in the 'resources/' directory")
        print("2. Run the demo notebook: demo_part3.ipynb")
        print("3. Or use the gradio demo: python -m openvoice_app --share")
        return True
    else:
        print("❌ Some tests failed. Please check the installation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
