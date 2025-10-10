README - Chess policy value (progetto didattico)
===============================================

IT (Italiano)
-------------
Scopo:
  Progetto didattico per esplorare un modello policy+value per scacchi:
  conversione FEN→planes, pipeline di dataset e allenamento con masking
  delle mosse legali. L'obiettivo principale è l'apprendimento pratico
  di tecniche di preprocessing, TensorFlow/Keras e rappresentazione delle mosse.

Contenuto della repository:
  - script Python (.py) per estrazione, conversione, costruzione e training
  - esempi di log (log.txt), file di dataset (.csv, .npz) e dump PGN compressi (.pgn.zst)

Dati:
  I dati di partenza (PGN) sono basati su dump pubblici Lichess (lichess.org).
  Questa repository può contenere versioni processate dei dati; i dump originali
  non sono distribuiti qui per ragioni di dimensione/licenza.

Riconoscimenti:
  Questo lavoro è stato sviluppato dall'autore del repository con *assistenza
  significativa* del modello linguistico ChatGPT (OpenAI) per la generazione,
  il refactoring e i suggerimenti di codice, nonché questo stesso file. 
  Ho incluso questa nota per trasparenza e correttezza nella paternità del codice.

EN (English)
------------
Purpose:
  Educational project exploring a chess policy+value model: FEN→planes conversion,
  data pipeline and training with masking of legal moves. The main goal is hands-on
  learning of preprocessing, TensorFlow/Keras and move representation.

Repository contents:
  - Python scripts (.py) for extraction, conversion, model building and training
  - sample logs (log.txt), dataset files (.csv, .npz) and compressed PGN dumps (.pgn.zst)

Data:
  Source PGN data is based on public Lichess dumps (lichess.org). The repo may
  include processed dataset files; original dumps are not redistributed here.

Acknowledgements:
  This work was developed by the repository author with substantial assistance
  from the ChatGPT language model (OpenAI) for code generation, refactoring and
  suggestions, as well as for this very file.
  This note is provided for transparency in authorship.