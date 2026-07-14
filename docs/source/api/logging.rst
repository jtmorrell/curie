.. _logging_api:

=======
Logging
=======

Curie reports what its fits did through console messages of the form
``[LEVEL] Class.method: message`` (see :ref:`methods_reporting` for the
conventions).  Three functions control the output:

.. autofunction:: curie.set_log_level

.. autofunction:: curie.quiet_warnings

.. autofunction:: curie.log_to
