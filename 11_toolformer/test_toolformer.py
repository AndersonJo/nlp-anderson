#!/usr/bin/env python3
"""
Comprehensive tests for Toolformer implementation.
Tests all major components and functionality.
"""

import unittest
import torch
import tempfile
import os
import shutil
from typing import List, Dict

from toolformer import ToolformerModel, ToolformerConfig, APICallEncoder
from tools import Calculator, QASystem, Calendar, Translator, get_all_tools, ToolRegistry
from training import ToolformerTrainer, TrainingConfig, ToolformerDataset
from loss import ToolformerLoss, ToolformerMetrics
from inference import ToolformerInference
from main import ToolformerTrainingPipeline


class TestAPICallEncoder(unittest.TestCase):
    """Test API call encoding and decoding"""
    
    def setUp(self):
        self.config = ToolformerConfig()
        self.encoder = APICallEncoder(self.config)
    
    def test_encode_call(self):
        """Test basic API call encoding"""
        result = self.encoder.encode_call("Calculator", "2+3")
        expected = "<API> Calculator(2+3) </API>"
        self.assertEqual(result, expected)
    
    def test_encode_call_with_result(self):
        """Test API call encoding with result"""
        result = self.encoder.encode_call_with_result("Calculator", "2+3", "5")
        expected = "<API> Calculator(2+3) → 5 </API>"
        self.assertEqual(result, expected)
    
    def test_decode_calls(self):
        """Test API call decoding"""
        text = "The result is <API> Calculator(2+3) → 5 </API> which is correct."
        calls = self.encoder.decode_calls(text)
        
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['tool_name'], 'Calculator')
        self.assertEqual(calls[0]['input'], '2+3')
        self.assertEqual(calls[0]['result'], '5')
    
    def test_decode_calls_without_result(self):
        """Test decoding calls without results"""
        text = "Calculate <API> Calculator(10*5) </API> for me."
        calls = self.encoder.decode_calls(text)
        
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['tool_name'], 'Calculator')
        self.assertEqual(calls[0]['input'], '10*5')
        self.assertIsNone(calls[0]['result'])
    
    def test_remove_results(self):
        """Test removing results from API calls"""
        text = "The answer is <API> Calculator(5*6) → 30 </API> obviously."
        result = self.encoder.remove_results(text)
        expected = "The answer is <API> Calculator(5*6) </API> obviously."
        self.assertEqual(result, expected)


class TestTools(unittest.TestCase):
    """Test individual tools"""
    
    def setUp(self):
        self.calculator = Calculator()
        self.qa = QASystem()
        self.calendar = Calendar()
        self.translator = Translator()
    
    def test_calculator_basic_operations(self):
        """Test calculator basic operations"""
        self.assertEqual(self.calculator.execute("2+3"), "5")
        self.assertEqual(self.calculator.execute("10-4"), "6")
        self.assertEqual(self.calculator.execute("3*7"), "21")
        self.assertEqual(self.calculator.execute("15/3"), "5")
    
    def test_calculator_complex_operations(self):
        """Test calculator complex operations"""
        self.assertEqual(self.calculator.execute("2+3*4"), "14")
        self.assertEqual(self.calculator.execute("(2+3)*4"), "20")
        self.assertTrue(self.calculator.execute("10/0").startswith("Error"))
    
    def test_calculator_invalid_input(self):
        """Test calculator with invalid input"""
        result = self.calculator.execute("invalid")
        self.assertTrue(result.startswith("Error"))
    
    def test_qa_system(self):
        """Test QA system"""
        result = self.qa.execute("What is the capital of France?")
        self.assertEqual(result, "Paris")
        
        result = self.qa.execute("Unknown question")
        self.assertTrue(result.startswith("I don't have information"))
    
    def test_calendar_today(self):
        """Test calendar today function"""
        result = self.calendar.execute("today")
        self.assertRegex(result, r'\d{4}-\d{2}-\d{2}')
    
    def test_translator(self):
        """Test translator"""
        result = self.translator.execute("hello to spanish")
        self.assertEqual(result, "hola")
        
        result = self.translator.execute("invalid format")
        self.assertTrue(result.startswith("Error"))


class TestToolRegistry(unittest.TestCase):
    """Test tool registry functionality"""
    
    def setUp(self):
        self.registry = ToolRegistry()
    
    def test_tool_registration(self):
        """Test tool registration"""
        tools = self.registry.list_tools()
        self.assertIn("Calculator", tools)
        self.assertIn("QA", tools)
    
    def test_get_tool(self):
        """Test getting tools by name"""
        calc = self.registry.get_tool("Calculator")
        self.assertIsNotNone(calc)
        self.assertEqual(calc.name(), "Calculator")
    
    def test_execute_tool(self):
        """Test tool execution through registry"""
        result = self.registry.execute_tool("Calculator", "5+7")
        self.assertEqual(result, "12")
    
    def test_tool_descriptions(self):
        """Test tool descriptions"""
        descriptions = self.registry.get_tool_descriptions()
        self.assertIn("Calculator", descriptions)
        self.assertTrue(len(descriptions["Calculator"]) > 0)


class TestToolformerModel(unittest.TestCase):
    """Test Toolformer model functionality"""
    
    def setUp(self):
        self.config = ToolformerConfig(
            n_embd=128,  # Smaller for testing
            n_layer=2,
            n_head=2,
            n_positions=256
        )
        self.tools = [Calculator(), QASystem()]
        self.model = ToolformerModel(self.config, self.tools)
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model)
        self.assertEqual(len(self.model.tools), 2)
        self.assertIn("Calculator", self.model.tools)
    
    def test_forward_pass(self):
        """Test forward pass"""
        input_text = "Calculate 2+3"
        input_ids = self.model.tokenizer.encode(input_text, return_tensors='pt')
        
        outputs = self.model(input_ids)
        
        self.assertIn('logits', outputs)
        self.assertIn('loss', outputs)
        self.assertEqual(outputs['logits'].shape[0], 1)  # batch size
        self.assertEqual(outputs['logits'].shape[1], input_ids.shape[1])  # sequence length
    
    def test_api_start_probabilities(self):
        """Test API start probability computation"""
        input_text = "What is 2+3?"
        input_ids = self.model.tokenizer.encode(input_text, return_tensors='pt')
        
        probs = self.model.compute_api_start_probabilities(input_ids)
        
        self.assertEqual(probs.shape, input_ids.shape)
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))
    
    def test_sample_api_candidates(self):
        """Test API candidate sampling"""
        text = "What is the result of 5 plus 7?"
        candidates = self.model.sample_api_candidates(text, top_k=3)
        
        self.assertIsInstance(candidates, list)
        self.assertLessEqual(len(candidates), 3)
    
    def test_execute_tool(self):
        """Test tool execution"""
        result = self.model.execute_tool("Calculator", "3*4")
        self.assertEqual(result, "12")
        
        result = self.model.execute_tool("NonexistentTool", "test")
        self.assertTrue(result.startswith("Error"))


class TestToolformerLoss(unittest.TestCase):
    """Test Toolformer loss function"""
    
    def setUp(self):
        self.config = ToolformerConfig(
            n_embd=64,  # Very small for testing
            n_layer=1,
            n_head=1,
            n_positions=128
        )
        self.tools = [Calculator()]
        self.model = ToolformerModel(self.config, self.tools)
        self.loss_fn = ToolformerLoss(self.model)
    
    def test_language_modeling_loss(self):
        """Test language modeling loss computation"""
        text = "Hello world"
        input_ids = self.model.tokenizer.encode(text, return_tensors='pt')
        attention_mask = torch.ones_like(input_ids)
        
        loss_dict = self.loss_fn(input_ids, attention_mask, input_ids)
        
        self.assertIn('loss', loss_dict)
        self.assertIn('language_modeling_loss', loss_dict)
        self.assertTrue(loss_dict['loss'] >= 0)
    
    def test_loss_improvement_computation(self):
        """Test loss improvement computation"""
        original_text = "The answer is 5."
        augmented_text = "The answer is <API> Calculator(2+3) → 5 </API>."
        
        improvement = self.loss_fn.compute_loss_improvement(original_text, augmented_text)
        self.assertIsInstance(improvement, float)


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline"""
    
    def setUp(self):
        self.config = ToolformerConfig(
            n_embd=32,  # Very small for testing
            n_layer=1,
            n_head=1,
            n_positions=64
        )
        self.training_config = TrainingConfig(
            batch_size=1,
            num_epochs=1,
            learning_rate=1e-4
        )
        self.tools = [Calculator()]
        self.model = ToolformerModel(self.config, self.tools)
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test dataset creation"""
        texts = ["What is 2+3?", "Calculate 5*7."]
        dataset = ToolformerDataset(texts, self.model, self.training_config, is_training=True)
        
        self.assertGreaterEqual(len(dataset), len(texts))
        
        sample = dataset[0]
        self.assertIn('input_ids', sample)
        self.assertIn('attention_mask', sample)
        self.assertIn('labels', sample)
    
    def test_training_step(self):
        """Test single training step"""
        trainer = ToolformerTrainer(self.model, self.training_config)
        
        batch = {
            'input_ids': torch.randint(0, 1000, (1, 10)),
            'attention_mask': torch.ones(1, 10),
            'labels': torch.randint(0, 1000, (1, 10))
        }
        
        loss = trainer._training_step(batch)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


class TestInference(unittest.TestCase):
    """Test inference functionality"""
    
    def setUp(self):
        self.config = ToolformerConfig(
            n_embd=64,
            n_layer=1,
            n_head=1,
            n_positions=128
        )
        self.tools = [Calculator(), QASystem()]
        self.model = ToolformerModel(self.config, self.tools)
        self.inference = ToolformerInference(self.model, temperature=1.0, max_length=50)
    
    def test_basic_generation(self):
        """Test basic text generation"""
        prompt = "Hello"
        results = self.inference.generate(prompt, num_return_sequences=1)
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], str)
        self.assertTrue(len(results[0]) > len(prompt))
    
    def test_tool_execution_in_text(self):
        """Test tool execution in text"""
        text = "The result is <API> Calculator(3+4) → 7 </API>."
        completed_text = self.inference.complete_with_tools(text)
        
        self.assertIn("7", completed_text)
    
    def test_explain_tool_usage(self):
        """Test tool usage explanation"""
        text = "Result: <API> Calculator(2*5) → 10 </API>"
        usage = self.inference.explain_tool_usage(text)
        
        self.assertIn("Calculator", usage)
        self.assertEqual(len(usage["Calculator"]), 1)
        self.assertEqual(usage["Calculator"][0]["input"], "2*5")
        self.assertEqual(usage["Calculator"][0]["result"], "10")


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        self.config = ToolformerConfig(
            n_embd=64,
            n_layer=1,
            n_head=1,
            n_positions=128
        )
        self.training_config = TrainingConfig(
            batch_size=1,
            num_epochs=1,
            learning_rate=1e-4
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline"""
        pipeline = ToolformerTrainingPipeline(self.config, self.training_config)
        
        train_data = ["What is 2+3?", "Calculate 4*5."]
        
        pipeline.train(train_data)
        
        prompt = "What is 1+1?"
        result = pipeline.generate_text(prompt, max_length=30)
        
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > len(prompt))
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        pipeline = ToolformerTrainingPipeline(self.config, self.training_config)
        
        save_path = os.path.join(self.temp_dir, "test_model.pt")
        pipeline.save_model(save_path)
        
        self.assertTrue(os.path.exists(save_path))
        
        new_pipeline = ToolformerTrainingPipeline(self.config, self.training_config)
        new_pipeline.load_model(save_path)
        
        self.assertIsNotNone(new_pipeline.model)


class TestMetrics(unittest.TestCase):
    """Test metrics tracking"""
    
    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = ToolformerMetrics()
        
        self.assertEqual(metrics.total_loss, 0.0)
        self.assertEqual(metrics.num_batches, 0)
    
    def test_metrics_update(self):
        """Test metrics update"""
        metrics = ToolformerMetrics()
        
        loss_dict = {
            'loss': torch.tensor(1.5),
            'language_modeling_loss': torch.tensor(1.2),
            'api_loss': torch.tensor(0.3),
            'perplexity': torch.tensor(4.5),
            'beneficial_calls': 2,
            'total_calls': 3
        }
        
        metrics.update(loss_dict)
        
        self.assertEqual(metrics.num_batches, 1)
        self.assertEqual(metrics.total_loss, 1.5)
        self.assertEqual(metrics.beneficial_calls, 2)
    
    def test_metrics_averages(self):
        """Test metrics averages computation"""
        metrics = ToolformerMetrics()
        
        for i in range(3):
            loss_dict = {
                'loss': torch.tensor(1.0 + i),
                'language_modeling_loss': torch.tensor(0.8 + i),
                'api_loss': torch.tensor(0.2),
                'perplexity': torch.tensor(3.0),
                'beneficial_calls': i + 1,
                'total_calls': 2
            }
            metrics.update(loss_dict)
        
        averages = metrics.compute_averages()
        
        self.assertEqual(averages['avg_total_loss'], 2.0)  # (1+2+3)/3
        self.assertEqual(averages['total_beneficial_calls'], 6)  # 1+2+3


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestAPICallEncoder,
        TestTools,
        TestToolRegistry,
        TestToolformerModel,
        TestToolformerLoss,
        TestTrainingPipeline,
        TestInference,
        TestIntegration,
        TestMetrics
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    print("Running Toolformer Tests")
    print("=" * 50)
    
    result = run_tests()
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")