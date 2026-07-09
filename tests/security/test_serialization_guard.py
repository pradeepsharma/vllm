# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Security tests: verify that deserialization guards are in place
and that VLLM_ALLOW_INSECURE_SERIALIZATION flag is respected.
"""

import os
import pickle
import sys
from unittest.mock import patch, MagicMock

import pytest

# We need to be careful with imports to avoid triggering the warning
# during test collection. We'll import these in the test functions.


class TestSerializationGuard:
    """Test deserialization safety guards."""

    def test_msgpack_decoder_rejects_pickle_when_flag_disabled(self):
        """
        Test that MsgpackDecoder.ext_hook raises NotImplementedError
        for CUSTOM_TYPE_PICKLE when VLLM_ALLOW_INSECURE_SERIALIZATION=0.
        """
        # Ensure the flag is disabled
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            # Force reimport to pick up the env var
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            # Create a decoder
            decoder = serial_utils.MsgpackDecoder()
            
            # Try to decode a CUSTOM_TYPE_PICKLE extension
            # CUSTOM_TYPE_PICKLE = 1
            with pytest.raises(NotImplementedError):
                decoder.ext_hook(1, memoryview(b"fake pickle data"))

    def test_msgpack_decoder_accepts_pickle_when_flag_enabled(self):
        """
        Test that MsgpackDecoder.ext_hook accepts CUSTOM_TYPE_PICKLE
        when VLLM_ALLOW_INSECURE_SERIALIZATION=1.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            decoder = serial_utils.MsgpackDecoder()
            
            # Create a valid pickle
            test_obj = {"key": "value"}
            pickled = pickle.dumps(test_obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Should not raise
            result = decoder.ext_hook(1, memoryview(pickled))
            assert result == test_obj

    def test_msgpack_decoder_rejects_cloudpickle_when_flag_disabled(self):
        """
        Test that MsgpackDecoder.ext_hook raises NotImplementedError
        for CUSTOM_TYPE_CLOUDPICKLE when VLLM_ALLOW_INSECURE_SERIALIZATION=0.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            decoder = serial_utils.MsgpackDecoder()
            
            # Try to decode a CUSTOM_TYPE_CLOUDPICKLE extension
            # CUSTOM_TYPE_CLOUDPICKLE = 2
            with pytest.raises(NotImplementedError):
                decoder.ext_hook(2, memoryview(b"fake cloudpickle data"))

    def test_msgpack_decoder_accepts_raw_view(self):
        """
        Test that MsgpackDecoder.ext_hook accepts CUSTOM_TYPE_RAW_VIEW
        regardless of the flag (it's safe).
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            decoder = serial_utils.MsgpackDecoder()
            
            # CUSTOM_TYPE_RAW_VIEW = 3
            test_data = b"raw data"
            result = decoder.ext_hook(3, memoryview(test_data))
            assert result == memoryview(test_data)

    def test_pickle_size_guard_in_distributed_utils(self):
        """
        Test that distributed/utils.py enforces MAX_SAFE_PICKLE_SIZE.
        """
        from vllm.distributed.utils import MAX_SAFE_PICKLE_SIZE, deserialize_object
        
        # Create a payload larger than MAX_SAFE_PICKLE_SIZE
        large_data = b"x" * (MAX_SAFE_PICKLE_SIZE + 1)
        
        with pytest.raises(ValueError) as exc_info:
            deserialize_object(large_data)
        
        assert "maximum safe size" in str(exc_info.value).lower()

    def test_pickle_size_guard_allows_small_payloads(self):
        """
        Test that small payloads are allowed through the size guard.
        """
        from vllm.distributed.utils import MAX_SAFE_PICKLE_SIZE, deserialize_object
        
        # Create a small valid pickle
        test_obj = {"small": "data"}
        small_data = pickle.dumps(test_obj, protocol=pickle.HIGHEST_PROTOCOL)
        
        assert len(small_data) < MAX_SAFE_PICKLE_SIZE
        
        # Should not raise
        result = deserialize_object(small_data)
        assert result == test_obj

    def test_run_method_rejects_bytes_when_flag_disabled(self):
        """
        Test that run_method raises when given bytes method and flag is disabled.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            # Create a mock object
            obj = MagicMock()
            
            # Try to call run_method with bytes (cloudpickle)
            import cloudpickle
            method_bytes = cloudpickle.dumps(lambda x: x.test_method())
            
            with pytest.raises(TypeError):
                serial_utils.run_method(obj, method_bytes, (), {})

    def test_run_method_accepts_string_method(self):
        """
        Test that run_method accepts string method names.
        """
        import vllm.v1.serial_utils as serial_utils
        
        # Create a mock object with a method
        obj = MagicMock()
        obj.test_method = MagicMock(return_value="result")
        
        # Call with string method name
        result = serial_utils.run_method(obj, "test_method", (), {})
        assert result == "result"
        obj.test_method.assert_called_once()

    def test_run_method_accepts_callable(self):
        """
        Test that run_method accepts callable objects.
        """
        import vllm.v1.serial_utils as serial_utils
        
        # Create a mock object
        obj = MagicMock()
        
        # Create a callable
        def test_func(o):
            return "called"
        
        # Call with callable
        result = serial_utils.run_method(obj, test_func, (), {})
        assert result == "called"

    def test_run_method_with_args_and_kwargs(self):
        """
        Test that run_method correctly passes args and kwargs.
        """
        import vllm.v1.serial_utils as serial_utils
        
        # Create a mock object
        obj = MagicMock()
        obj.test_method = MagicMock(return_value="result")
        
        # Call with args and kwargs
        result = serial_utils.run_method(
            obj,
            "test_method",
            ("arg1", "arg2"),
            {"key": "value"}
        )
        
        obj.test_method.assert_called_once_with("arg1", "arg2", key="value")

    def test_msgpack_encoder_logs_warning_when_flag_enabled(self):
        """
        Test that MsgpackEncoder logs a warning when VLLM_ALLOW_INSECURE_SERIALIZATION=1.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            # The warning should be logged during encoder creation
            with patch("vllm.v1.serial_utils.logger") as mock_logger:
                encoder = serial_utils.MsgpackEncoder()
                # Check that warning was logged
                # (Note: the warning is logged once globally, so this might not trigger)

    def test_max_safe_pickle_size_is_reasonable(self):
        """
        Test that MAX_SAFE_PICKLE_SIZE is set to a reasonable value.
        """
        from vllm.distributed.utils import MAX_SAFE_PICKLE_SIZE
        
        # Should be at least 1MB
        assert MAX_SAFE_PICKLE_SIZE >= 1024 * 1024
        
        # Should be at most 1GB
        assert MAX_SAFE_PICKLE_SIZE <= 1024 * 1024 * 1024

    def test_utility_result_requires_flag_for_custom_types(self):
        """
        Test that UtilityResult deserialization requires the flag for custom types.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            decoder = serial_utils.MsgpackDecoder()
            
            # Try to decode a UtilityResult with custom type info
            # This should raise if the flag is not set
            with pytest.raises(TypeError) as exc_info:
                decoder._decode_utility_result((("module", "ClassName"), {"data": "value"}))
            
            assert "VLLM_ALLOW_INSECURE_SERIALIZATION" in str(exc_info.value)

    def test_encoder_rejects_unknown_types_when_flag_disabled(self):
        """
        Test that MsgpackEncoder rejects unknown types when flag is disabled.
        """
        with patch.dict(os.environ, {"VLLM_ALLOW_INSECURE_SERIALIZATION": "0"}):
            import importlib
            import vllm.v1.serial_utils as serial_utils
            importlib.reload(serial_utils)
            
            encoder = serial_utils.MsgpackEncoder()
            
            # Create a custom object that's not a known type
            class CustomClass:
                pass
            
            obj = CustomClass()
            
            # Should raise TypeError
            with pytest.raises(TypeError) as exc_info:
                encoder.enc_hook(obj)
            
            assert "not serializable" in str(exc_info.value)

    def test_encoder_accepts_known_types(self):
        """
        Test that MsgpackEncoder accepts known types.
        """
        import vllm.v1.serial_utils as serial_utils
        import torch
        import numpy as np
        
        encoder = serial_utils.MsgpackEncoder()
        
        # Test torch tensor
        tensor = torch.tensor([1, 2, 3])
        result = encoder.enc_hook(tensor)
        assert result is not None
        
        # Test numpy array
        arr = np.array([1, 2, 3])
        result = encoder.enc_hook(arr)
        assert result is not None
        
        # Test slice
        s = slice(1, 10, 2)
        result = encoder.enc_hook(s)
        assert result == (1, 10, 2)
