#!/usr/bin/env python3
"""
Test script for official SeamlessStreaming agent initialization
"""

import sys
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_official_agent():
    """Test the official SeamlessStreaming agent initialization"""
    
    logger.info("🧪 Testing Official SeamlessStreaming Agent Initialization")
    logger.info("=" * 60)
    
    try:
        # Import the official translator
        logger.info("📦 Importing OfficialStreamingTranslator...")
        sys.path.append('backend')
        from streaming_server import OfficialStreamingTranslator
        
        # Initialize translator
        logger.info("🚀 Initializing translator...")
        translator = OfficialStreamingTranslator(source_lang="eng", target_lang="ben")
        
        logger.info("✅ SUCCESS: Official SeamlessStreaming agent initialized successfully!")
        logger.info(f"   📱 Agent type: {type(translator.agent).__name__}")
        logger.info(f"   🎯 Task: {translator.args.task}")
        logger.info(f"   💾 Device: {translator.args.device}")
        logger.info(f"   📊 Dtype: {translator.args.dtype}")
        
        # Test basic agent properties
        if hasattr(translator.agent, 'pipeline'):
            logger.info(f"   🔗 Pipeline: {len(translator.agent.pipeline)} agents")
            for i, agent_class in enumerate(translator.agent.pipeline):
                logger.info(f"      {i+1}. {agent_class.__name__}")
        
        return True
        
    except Exception as e:
        logger.error("❌ FAILED: Agent initialization failed")
        logger.error(f"   Error: {e}")
        logger.error(f"   Type: {type(e).__name__}")
        logger.error("\n🔍 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_official_agent()
    sys.exit(0 if success else 1)