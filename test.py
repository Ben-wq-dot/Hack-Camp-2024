import spacy
 
TRAIN_DATA = [
    ("Update XMLPayload.java insert newline into */package xmlwriter", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Added ICSE Score ADR", {"cats": {"ADD_FEATURE": 1.0, "REFACTOR_CODE": 0.0}}),
    ("Minor edits", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Remove unused imports", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Setup Logger and print XML to log file", {"cats": {"ADD_FEATURE": 1.0, "REFACTOR_CODE": 0.0}}),
    ("Create and test new logger infrastructure", {"cats": {"ADD_FEATURE": 1.0, "REFACTOR_CODE": 0.0}}),
    ("More commenting tidy-ups", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Tidy up comments relating to migration from Vector to ArrayList", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Tidy-up commenting around migration to ArrayList (from vector)", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("More commenting tidy-up", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Change visibility of abstract class constructors to protected", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Check return string for null", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Remove XML parser vulnerability to XXE attacks", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Move StringWriter into try-catch block", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Move FileReader inside try block", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Close FileReader in finally clause", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Create nested try-catch-finally block for buffered reader and file inputStream", {"cats": {"ADD_FEATURE": 1.0, "REFACTOR_CODE": 0.0}}),
    ("Add try...catch block for bufferedreader", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Avoid closing null streams", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Insert file handling into Try... Catch block", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Add @Override to hashCode()", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Add @Override annotation to equals()", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Rename the equals() method to prevent any confusion with Object.equals(Object)", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Fix sonar code execution after changing package name", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Disguise comment out source code", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Update sonar-project.properties", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Remove unused include", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Rename XMLWriter package to xmlwriter for naming convention compliance", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Reverted to using vector in specific cases", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Finishing migrating Vector to ArrayList classes", {"cats": {"ADD_FEATURE": 1.0, "REFACTOR_CODE": 0.0}}),
    ("Refactor vector to ArrayList types", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Tidy up a couple of comments", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Tidy up some comments", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Improve code quality by removing un-needed debug print statements", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Constructors of an 'abstract' class should not be declared 'public'", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Improve code quality by removing unused debug print statements", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Remove spaces from within paths to source and binary files", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("A further attempt to fix paths to source and binary files", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}}),
    ("Fixing paths to source and binaries", {"cats": {"ADD_FEATURE": 0.0, "REFACTOR_CODE": 1.0}})
]
 
from spacy.lookups import Lookups
from spacy.training.example import Example
# Load spaCy model
nlp = spacy.load("en_core_web_sm")
 
# Manually add the `lexeme_norm` lookup table
lookups = Lookups()
if not nlp.vocab.lookups.has_table("lexeme_norm"):
    lookups.add_table("lexeme_norm", {"example": "example_norm"})  # Example norm data
    nlp.vocab.lookups.add_table("lexeme_norm", lookups.get_table("lexeme_norm"))
 
# Proceed with your training setup
if "textcat" not in nlp.pipe_names:
    textcat = nlp.add_pipe("textcat", last=True)
else:
    textcat = nlp.get_pipe("textcat")
 
# Add labels
textcat.add_label("ADD_FEATURE")
textcat.add_label("REFACTOR_CODE")
 
 
 
# Training data preparation
examples = []
for text, annotations in TRAIN_DATA:
    doc = nlp.make_doc(text)
    examples.append(Example.from_dict(doc, annotations))
 
# Training
optimizer = nlp.begin_training()
 
for epoch in range(10):
    losses = {}
    nlp.update(examples, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch+1}, Loss: {losses['textcat']}")
 
# Save model
nlp.to_disk("text_categorizer_model")
 
 
# Load the model later
nlp = spacy.load("text_categorizer_model")