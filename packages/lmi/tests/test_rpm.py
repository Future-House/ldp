import asyncio
import time
from typing import List
import pytest
from lmi.llms import LiteLLMModel
from lmi.types import Message
from limits import RateLimitItemPerMinute
from lmi.rate_limiter import GLOBAL_LIMITER 


async def test_rpm_limit():
    """Test the request rate limit (RPM) functionality"""
    messages = [Message(role="user", content="Hi")]
    results = {}
    
    # Test request time without rate limit
    print("\n=== Testing without rate limit ===")
    model_no_limit = LiteLLMModel(
        name="deepseek/deepseek-chat",
        config={
            "model_list": [{
                "model_name": "deepseek/deepseek-chat",
                "litellm_params": {
                    "model": "deepseek/deepseek-chat",
                    "temperature": 0.1,
                    "max_tokens": 4096,
                }
            }]
        }
    )
    
    results["No Limit"] = {}
    for req_count in [2, 3, 4]:
        print(f"\nTesting time to complete {req_count} requests without limit...")
        start_time = time.time()
        
        for i in range(req_count):
            try:
                print(f"Sending unlimited request {i+1}/{req_count}...")
                await model_no_limit.call_single(messages)
                print(f"Unlimited request {i+1}/{req_count} completed successfully")
            except Exception as e:
                print(f"Unlimited request {i+1}/{req_count} failed: {e}")
        
        elapsed_time = time.time() - start_time
        results["No Limit"][req_count] = elapsed_time
        print(f"Time to complete {req_count} requests without limit: {elapsed_time:.2f} seconds")
        
        # Wait before next test round
        await asyncio.sleep(30)
    
    # Test with different RPM limits
    for rpm_limit in [1, 2, 3]:
        print(f"\n=== Testing with RPM={rpm_limit} ===")
        # Wait for limit reset
        print(f"Waiting 60 seconds for system reset...")
        await asyncio.sleep(60)
        
        model_with_limit = LiteLLMModel(
            name="deepseek/deepseek-chat",
            config={
                "model_list": [{
                    "model_name": "deepseek/deepseek-chat",
                    "litellm_params": {
                        "model": "deepseek/deepseek-chat",
                        "temperature": 0.1,
                        "max_tokens": 4096,
                    }
                }],
                "request_limit": {
                    "deepseek/deepseek-chat": f"{rpm_limit}/minute"
                }
            }
        )
        
        results[f"RPM={rpm_limit}"] = {}
        # Test time needed for RPM+1 requests
        req_count = rpm_limit + 1
        print(f"\nTesting time to complete {req_count} requests with RPM={rpm_limit}...")
        start_time = time.time()
        
        for i in range(req_count):
            try:
                print(f"Sending limited request {i+1}/{req_count}...")
                await model_with_limit.call_single(messages)
                print(f"Limited request {i+1}/{req_count} completed successfully")
            except Exception as e:
                print(f"Limited request {i+1}/{req_count} failed: {e}")
        
        elapsed_time = time.time() - start_time
        results[f"RPM={rpm_limit}"][req_count] = elapsed_time
        print(f"Time to complete {req_count} requests with RPM={rpm_limit}: {elapsed_time:.2f} seconds")
    
    # Summarize and analyze results
    print("\n\n=== Test Results Summary ===")
    print("Without rate limit:")
    for req_count, time_spent in results["No Limit"].items():
        print(f"  Time to complete {req_count} requests: {time_spent:.2f} seconds")
    
    for rpm_limit in [1, 2, 3]:
        req_count = rpm_limit + 1
        time_spent = results[f"RPM={rpm_limit}"].get(req_count, "Not completed")
        if isinstance(time_spent, (int, float)):
            time_spent = f"{time_spent:.2f} seconds"
        print(f"\nWith RPM={rpm_limit}:")
        print(f"  Time to complete {req_count} requests: {time_spent}")
    
    # Validate results
    for rpm_limit in [1, 2, 3]:
        req_count = rpm_limit + 1
        time_with_limit = results[f"RPM={rpm_limit}"].get(req_count)
        if isinstance(time_with_limit, (int, float)):
            # With limit, completing RPM+1 requests should take at least 60 seconds
            assert time_with_limit >= 60.0, f"With RPM={rpm_limit}, completing {req_count} requests should take at least 60 seconds, but only took {time_with_limit:.2f} seconds"
            # With limit should be slower than without limit
            time_no_limit = results["No Limit"].get(req_count)
            if isinstance(time_no_limit, (int, float)):
                assert time_with_limit > time_no_limit, f"With RPM={rpm_limit}, request time should be longer than without limit"
    
    
    status = await GLOBAL_LIMITER.rate_limit_status()
    print("\nCurrent limit status:", status)
    
    return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    asyncio.run(test_rpm_limit())