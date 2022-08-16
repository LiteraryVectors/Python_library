# To use your library from colab

```
#flush the old directory
!rm -r  'Python_library'

my_github_name = 'LiteraryVectors'

clone_url = f'https://github.com/{LiteraryVectors}/Python_library.git'

#this adds the library to colab so you can now import it
!git clone $clone_url

import Python_library as pl
```

# Test the import

`my.hello()`
