# test_rag.py
import unittest
from main import *
import time
import os
from langchain_core.messages import HumanMessage

# Set environment variable for testing
os.environ["USER_AGENT"] = "RAG_AI_Agent_Test/1.0 (Python; Educational_Purpose)"


def test_rag_system():
    """Comprehensive test of the RAG system"""
    # Initialize the agent
    agent = build_rag_agent()

    test_cases = [
        {
            "query": "What are the core features of AI agents and how do they achieve autonomy?",
            "expected_keywords": ["autonomous", "goal-oriented", "reasoning", "planning", "memory"]
        },
        {
            "query": "Explain chain-of-thought prompting and few-shot learning techniques",
            "expected_keywords": ["chain-of-thought", "few-shot", "prompting", "reasoning", "examples"]
        },
        {
            "query": "How does HNSW indexing work in vector databases?",
            "expected_keywords": ["HNSW", "indexing", "vector", "similarity", "search"]
        },
        {
            "query": "What are the best practices for RAG implementation?",
            "expected_keywords": ["RAG", "retrieval", "generation", "implementation", "best practices"]
        },
        {
            "query": "Describe transformer architecture and attention mechanisms",
            "expected_keywords": ["transformer", "attention", "architecture", "neural", "mechanism"]
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"TEST {i}: {test_case['query']}")
        print(f"{'=' * 80}")

        try:
            config = {"configurable": {"thread_id": f"test_{i}"}}

            # Invoke the agent with correct message format
            start_time = time.time()
            response = agent.invoke(
                {"messages": [HumanMessage(content=test_case['query'])]},
                config=config
            )
            end_time = time.time()

            # Extract response details from the correct structure
            final_response = response['messages'][-1].content
            documents_found = response.get('documents_found', False)
            route = response.get('route', 'unknown')

            print(f"\nRESPONSE ({len(final_response)} chars, {end_time - start_time:.2f}s):")
            print(final_response[:500] + "..." if len(final_response) > 500 else final_response)

            print(f"\nDOCUMENTS FOUND: {documents_found}")
            print(f"ROUTE TAKEN: {route}")

            # Evaluate keyword relevance
            found_keywords = []
            response_lower = final_response.lower()
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)

            relevance_score = len(found_keywords) / len(test_case['expected_keywords']) * 100

            print(f"EXPECTED KEYWORDS: {', '.join(test_case['expected_keywords'])}")
            print(f"FOUND KEYWORDS: {', '.join(found_keywords)}")
            print(f"RELEVANCE SCORE: {relevance_score:.1f}%")

            # Test result evaluation
            test_passed = (
                    len(final_response) >= 100 and  # Adjusted minimum length
                    relevance_score >= 20.0  # Adjusted relevance threshold
            )

            print(f"TEST RESULT: {'✅ PASS' if test_passed else '❌ FAIL'}")

            results.append({
                "test_id": i,
                "query": test_case['query'],
                "response_length": len(final_response),
                "relevance_score": relevance_score,
                "documents_found": documents_found,
                "route": route,
                "processing_time": end_time - start_time,
                "passed": test_passed
            })

        except Exception as e:
            print(f"ERROR: {str(e)}")
            results.append({
                "test_id": i,
                "query": test_case['query'],
                "error": str(e),
                "passed": False
            })

    # Summary Report
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY REPORT")
    print(f"{'=' * 80}")

    passed_tests = sum(1 for r in results if r.get('passed', False))
    total_tests = len(results)
    avg_relevance = sum(r.get('relevance_score', 0) for r in results) / total_tests
    avg_response_length = sum(r.get('response_length', 0) for r in results if 'response_length' in r) / len(
        [r for r in results if 'response_length' in r])
    avg_processing_time = sum(r.get('processing_time', 0) for r in results if 'processing_time' in r) / len(
        [r for r in results if 'processing_time' in r])

    print(f"Total tests: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Pass rate: {passed_tests / total_tests * 100:.1f}%")
    print(f"Average relevance score: {avg_relevance:.1f}%")
    print(f"Average response length: {avg_response_length:.0f} characters")
    print(f"Average processing time: {avg_processing_time:.2f} seconds")

    # Route distribution
    routes = [r.get('route', 'unknown') for r in results if 'route' in r]
    if routes:
        print(f"\nROUTE DISTRIBUTION:")
        route_counts = {}
        for route in routes:
            route_counts[route] = route_counts.get(route, 0) + 1
        for route, count in route_counts.items():
            print(f"  {route}: {count} times")

    return results


def test_document_loading():
    """Test document loading functionality"""
    print("\n" + "=" * 80)
    print("TESTING DOCUMENT LOADING")
    print("=" * 80)

    # Test loading default documents
    try:
        success = load_default_documents()
        print(f"Default document loading: {'✅ SUCCESS' if success else '❌ FAILED'}")

        # Check collection stats
        try:
            count = collection.count()
            print(f"Documents in collection: {count}")
        except Exception as e:
            print(f"Error getting collection stats: {e}")

    except Exception as e:
        print(f"Error loading default documents: {e}")


def test_text_processing():
    """Test text processing functions"""
    print("\n" + "=" * 80)
    print("TESTING TEXT PROCESSING FUNCTIONS")
    print("=" * 80)

    # Test text cleaning
    test_text = "  This is   a test   text with\n\nextra   whitespace!@#$%  "
    cleaned = clean_text(test_text)
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{cleaned}'")
    print(f"Text cleaning: {'✅ PASS' if len(cleaned) < len(test_text) else '❌ FAIL'}")

    # Test text chunking
    sample_text = "This is a sample text for testing chunking functionality. " * 20
    chunks = chunk_text(sample_text, chunk_size=50, overlap=10)
    print(f"\nChunking test:")
    print(f"Original text length: {len(sample_text.split())} words")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Text chunking: {'✅ PASS' if len(chunks) > 1 else '❌ FAIL'}")


if __name__ == "__main__":
    print("Initializing test environment...")

    # Initialize components
    try:
        embedding_model = load_embedding_model()
        collection = setup_vector_database()
        print("Components initialized successfully!")
    except Exception as e:
        print(f"Error initializing components: {e}")
        exit(1)

    # Test document loading
    test_document_loading()

    # Test text processing functions
    test_text_processing()

    # Run the main RAG system tests
    test_results = test_rag_system()

    # Overall test summary
    print(f"\n{'=' * 80}")
    print("OVERALL TEST SUMMARY")
    print(f"{'=' * 80}")

    total_passed = sum(1 for r in test_results if r.get('passed', False))
    total_tests = len(test_results)

    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"⚠️  {total_tests - total_passed} out of {total_tests} tests failed")

    print("Test execution completed.")
