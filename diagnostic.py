#!/usr/bin/env python
"""Diagnostic script to understand KeyNMF API."""

from turftopic import KeyNMF

print("Creating and fitting a small KeyNMF model for diagnosis...")
model = KeyNMF(n_components=3, random_state=42)
dummy_corpus = ['apple banana cherry fruit', 'dog cat bird animal', 'red blue green color']
model.fit(dummy_corpus)

print("\nTesting KeyNMF API methods:\n" + "="*60)

# Test get_topics()
print("\n1. model.get_topics():")
try:
    result = model.get_topics()
    print(f"   ✓ Works! Type: {type(result)}")
    if hasattr(result, 'keys'):
        print(f"     Keys: {list(result.keys())}")
        if result:
            first_key = list(result.keys())[0]
            print(f"     Sample (topic {first_key}): {result[first_key]}")
except AttributeError:
    print(f"   ✗ AttributeError - Method doesn't exist")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test get_topic_words()
print("\n2. model.get_topic_words():")
try:
    result = model.get_topic_words()
    print(f"   ✓ Works! Type: {type(result)}")
    print(f"     Shape/Length: {result.shape if hasattr(result, 'shape') else len(result)}")
    if hasattr(result, '__len__') and len(result) > 0:
        print(f"     Sample (topic 0): {result[0]}")
except AttributeError:
    print(f"   ✗ AttributeError - Method doesn't exist")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test top_words attribute
print("\n3. model.top_words:")
try:
    result = model.top_words
    print(f"   ✓ Works! Type: {type(result)}")
    print(f"     Shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
    print(f"     Sample (topic 0): {result[0] if hasattr(result, '__len__') else result}")
except AttributeError:
    print(f"   ✗ AttributeError - Attribute doesn't exist")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test vocabulary
print("\n4. model.vocabulary:")
try:
    result = model.vocabulary
    print(f"   ✓ Works! Type: {type(result)}")
    print(f"     Length: {len(result)}")
    print(f"     Sample: {result[:5]}")
except AttributeError:
    print(f"   ✗ AttributeError - Attribute doesn't exist")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

# Test topics
print("\n5. model.topics:")
try:
    result = model.topics
    print(f"   ✓ Works! Type: {type(result)}")
    print(f"     Shape: {result.shape if hasattr(result, 'shape') else len(result)}")
except AttributeError:
    print(f"   ✗ AttributeError - Attribute doesn't exist")
except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

print("\n" + "="*60)
print("Summary of what works:")
works = []
for method in ['get_topics', 'get_topic_words', 'top_words', 'vocabulary', 'topics']:
    try:
        if method.startswith('get_'):
            getattr(model, method)()
        else:
            getattr(model, method)
        works.append(method)
    except:
        pass

if works:
    print(f"✓ Working: {', '.join(works)}")
else:
    print("✗ None of the standard methods work!")
