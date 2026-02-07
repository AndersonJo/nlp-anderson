import re
import requests
import datetime
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import json
import math
import calendar


class Tool(ABC):
    """Abstract base class for all tools"""
    
    @abstractmethod
    def name(self) -> str:
        """Return the tool name"""
        pass
    
    @abstractmethod
    def execute(self, input_str: str) -> str:
        """Execute the tool with given input and return result"""
        pass
    
    @abstractmethod
    def description(self) -> str:
        """Return tool description for prompting"""
        pass


class Calculator(Tool):
    """Calculator tool for arithmetic operations - exact implementation from paper"""
    
    def name(self) -> str:
        return "Calculator"
    
    def execute(self, input_str: str) -> str:
        """Execute mathematical calculations"""
        try:
            expression = input_str.strip()
            
            allowed_chars = set('0123456789+-*/().% ')
            if not all(c in allowed_chars for c in expression):
                return "Error: Invalid characters in expression"
            
            if any(keyword in expression.lower() for keyword in ['import', 'exec', 'eval', '__']):
                return "Error: Unsafe expression"
            
            result = eval(expression)
            
            if isinstance(result, float):
                if result.is_integer():
                    return str(int(result))
                else:
                    return f"{result:.6f}".rstrip('0').rstrip('.')
            
            return str(result)
            
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: Invalid calculation - {str(e)}"
    
    def description(self) -> str:
        return "Calculator(expression) - Evaluates mathematical expressions like '2+3', '10*5', 'sqrt(16)'"


class QASystem(Tool):
    """Question Answering system tool - simplified implementation"""
    
    def __init__(self):
        self.knowledge_base = {
            "what is the capital of france": "Paris",
            "what is the capital of germany": "Berlin", 
            "what is the capital of italy": "Rome",
            "what is the capital of spain": "Madrid",
            "what is the capital of uk": "London",
            "what is the capital of usa": "Washington D.C.",
            "what is the capital of japan": "Tokyo",
            "what is the capital of china": "Beijing",
            "what is the largest ocean": "Pacific Ocean",
            "what is the smallest continent": "Australia",
            "what is the highest mountain": "Mount Everest",
            "what is the longest river": "Nile River",
            "what is the speed of light": "299,792,458 meters per second",
            "what is the boiling point of water": "100 degrees Celsius",
            "what is the freezing point of water": "0 degrees Celsius",
        }
    
    def name(self) -> str:
        return "QA"
    
    def execute(self, input_str: str) -> str:
        """Answer questions using knowledge base"""
        question = input_str.strip().lower()
        
        for kb_question, answer in self.knowledge_base.items():
            if self._questions_match(question, kb_question):
                return answer
        
        return f"I don't have information about: {input_str}"
    
    def _questions_match(self, q1: str, q2: str) -> bool:
        """Check if questions match (simple keyword matching)"""
        q1_words = set(q1.split())
        q2_words = set(q2.split())
        
        common_words = q1_words.intersection(q2_words)
        return len(common_words) >= min(3, len(q2_words) - 1)
    
    def description(self) -> str:
        return "QA(question) - Answers factual questions using knowledge base"


class SearchEngine(Tool):
    """Search engine tool - simplified mock implementation"""
    
    def __init__(self):
        self.search_results = {
            "python": "Python is a high-level programming language known for its simplicity and readability.",
            "machine learning": "Machine learning is a method of data analysis that automates analytical model building.",
            "artificial intelligence": "Artificial intelligence refers to the simulation of human intelligence in machines.",
            "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns.",
            "renewable energy": "Renewable energy comes from sources that are naturally replenished, like solar and wind.",
            "quantum computing": "Quantum computing uses quantum mechanics to process information in fundamentally new ways.",
            "blockchain": "Blockchain is a distributed ledger technology that maintains a growing list of records.",
            "neural networks": "Neural networks are computing systems inspired by biological neural networks.",
        }
    
    def name(self) -> str:
        return "Search"
    
    def execute(self, input_str: str) -> str:
        """Search for information"""
        query = input_str.strip().lower()
        
        for topic, description in self.search_results.items():
            if topic in query or any(word in query for word in topic.split()):
                return description
        
        return f"No search results found for: {input_str}"
    
    def description(self) -> str:
        return "Search(query) - Searches for information about topics"


class Calendar(Tool):
    """Calendar tool for date and time operations"""
    
    def name(self) -> str:
        return "Calendar"
    
    def execute(self, input_str: str) -> str:
        """Execute calendar operations"""
        try:
            query = input_str.strip().lower()
            
            if "today" in query or "current date" in query:
                return datetime.date.today().strftime("%Y-%m-%d")
            
            elif "current time" in query or "time now" in query:
                return datetime.datetime.now().strftime("%H:%M:%S")
            
            elif "day of week" in query:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', input_str)
                if date_match:
                    date_str = date_match.group(1)
                    date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    return date_obj.strftime("%A")
                else:
                    return datetime.date.today().strftime("%A")
            
            elif "days until" in query:
                date_match = re.search(r'(\d{4}-\d{2}-\d{2})', input_str)
                if date_match:
                    target_date = datetime.datetime.strptime(date_match.group(1), "%Y-%m-%d").date()
                    today = datetime.date.today()
                    diff = (target_date - today).days
                    return str(diff)
            
            elif "days between" in query:
                dates = re.findall(r'(\d{4}-\d{2}-\d{2})', input_str)
                if len(dates) >= 2:
                    date1 = datetime.datetime.strptime(dates[0], "%Y-%m-%d").date()
                    date2 = datetime.datetime.strptime(dates[1], "%Y-%m-%d").date()
                    diff = abs((date2 - date1).days)
                    return str(diff)
            
            return "Error: Cannot understand calendar query"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def description(self) -> str:
        return "Calendar(query) - Handles date/time queries like 'today', 'day of week for 2023-01-01', 'days until 2023-12-25'"


class Translator(Tool):
    """Translation tool - simplified mock implementation"""
    
    def __init__(self):
        self.translations = {
            ("hello", "spanish"): "hola",
            ("hello", "french"): "bonjour", 
            ("hello", "german"): "guten tag",
            ("hello", "italian"): "ciao",
            ("thank you", "spanish"): "gracias",
            ("thank you", "french"): "merci",
            ("thank you", "german"): "danke",
            ("thank you", "italian"): "grazie",
            ("goodbye", "spanish"): "adiós",
            ("goodbye", "french"): "au revoir",
            ("goodbye", "german"): "auf wiedersehen", 
            ("goodbye", "italian"): "arrivederci",
            ("yes", "spanish"): "sí",
            ("yes", "french"): "oui",
            ("yes", "german"): "ja",
            ("yes", "italian"): "sì",
            ("no", "spanish"): "no",
            ("no", "french"): "non",
            ("no", "german"): "nein",
            ("no", "italian"): "no",
        }
    
    def name(self) -> str:
        return "Translator"
    
    def execute(self, input_str: str) -> str:
        """Translate text between languages"""
        try:
            parts = input_str.split(" to ")
            if len(parts) != 2:
                return "Error: Format should be 'text to language'"
            
            text, target_lang = parts[0].strip().lower(), parts[1].strip().lower()
            
            if (text, target_lang) in self.translations:
                return self.translations[(text, target_lang)]
            
            return f"Translation not available for '{text}' to {target_lang}"
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def description(self) -> str:
        return "Translator(text to language) - Translates text between languages, e.g. 'hello to spanish'"


class WikipediaSearch(Tool):
    """Wikipedia search tool - mock implementation"""
    
    def __init__(self):
        self.wiki_data = {
            "albert einstein": "Albert Einstein (1879-1955) was a theoretical physicist who developed the theory of relativity.",
            "paris": "Paris is the capital and most populous city of France, with an area of 105 square kilometres.",
            "python programming": "Python is an interpreted, high-level programming language with dynamic semantics.",
            "machine learning": "Machine learning is a method of data analysis that automates analytical model building.",
            "world war ii": "World War II was a global war that lasted from 1939 to 1945.",
            "photosynthesis": "Photosynthesis is the process by which plants use sunlight to synthesize foods from carbon dioxide and water.",
            "shakespeare": "William Shakespeare was an English playwright, poet, and actor, widely regarded as the greatest writer in the English language.",
        }
    
    def name(self) -> str:
        return "Wikipedia"
    
    def execute(self, input_str: str) -> str:
        """Search Wikipedia for information"""
        query = input_str.strip().lower()
        
        for topic, content in self.wiki_data.items():
            if topic in query or any(word in query for word in topic.split()):
                return content
        
        return f"No Wikipedia entry found for: {input_str}"
    
    def description(self) -> str:
        return "Wikipedia(query) - Searches Wikipedia for information about topics"


def get_all_tools() -> List[Tool]:
    """Return all available tools"""
    return [
        Calculator(),
        QASystem(), 
        SearchEngine(),
        Calendar(),
        Translator(),
        WikipediaSearch()
    ]


def get_tool_by_name(name: str) -> Optional[Tool]:
    """Get a specific tool by name"""
    tools = get_all_tools()
    for tool in tools:
        if tool.name().lower() == name.lower():
            return tool
    return None


class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        self.tools = {tool.name(): tool for tool in get_all_tools()}
    
    def register_tool(self, tool: Tool):
        """Register a new tool"""
        self.tools[tool.name()] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get tool by name"""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all available tool names"""
        return list(self.tools.keys())
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all tools"""
        return {name: tool.description() for name, tool in self.tools.items()}
    
    def execute_tool(self, name: str, input_str: str) -> str:
        """Execute a tool by name"""
        tool = self.get_tool(name)
        if tool:
            return tool.execute(input_str)
        return f"Error: Tool '{name}' not found"


if __name__ == "__main__":
    registry = ToolRegistry()
    
    print("Available tools:")
    for name, description in registry.get_tool_descriptions().items():
        print(f"- {name}: {description}")
    
    print("\nTesting tools:")
    
    print(f"Calculator(2+3*4): {registry.execute_tool('Calculator', '2+3*4')}")
    print(f"QA(What is the capital of France?): {registry.execute_tool('QA', 'What is the capital of France?')}")
    print(f"Calendar(today): {registry.execute_tool('Calendar', 'today')}")
    print(f"Translator(hello to spanish): {registry.execute_tool('Translator', 'hello to spanish')}")
    print(f"Wikipedia(Albert Einstein): {registry.execute_tool('Wikipedia', 'Albert Einstein')}")
    print(f"Search(python): {registry.execute_tool('Search', 'python')}")