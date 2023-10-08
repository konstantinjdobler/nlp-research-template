# dlib - the lib for deep learning research

`dlib` is a collection of tools, utility functions, and modules that I have found useful for deep learning research, mostly biased towards training LLMs.

## Installation

The easiest way to use `dlib` is to simply copy all files into a root-level `dlib/` folder in your project. The requirements are listed in `requiremnts.txt` but most of these are already present when doing NLP with `torch`. I have listed the "unusual" ones in `unusual-requirements.txt`.

Alternatively, you can use `git subrepo` or `git subtree` to clone all files inside a root-level `dlib` folder and be able to easily pull (and push) new updates.



## `git subrepo` (Preferred)
[`git subrepo`](https://github.com/ingydotnet/git-subrepo) fixes some peculiar "features" in `git subtree`, however you need to install the command.

### Installation

You need to install `git subrepo` once to add `dlib` to your repo.

```bash
git clone https://github.com/ingydotnet/git-subrepo ~/git-subrepo
echo 'source ~/git-subrepo/.rc' >> ~/.bashrc
```

or on MacOS:

```zsh
brew install git-subrepo
```

### Usage

Cloning:

```bash
git subrepo clone https://github.com/konstantinjdobler/dlib.git dlib
```

Pulling:

```bash
git subrepo pull dlib
```

Pushing. It's best to not mix changes in a subtree and the host repo in a single commit:

```bash
git subrepo push dlib
```


## `git subtree`

Here are some commands that are useful when using `git subtree`.

Cloning `dlib` into your existing repo:

```bash
git subtree add --prefix dlib/ https://github.com/konstantinjdobler/dlib.git main --squash
```


Pulling new commits:

```bash
git subtree pull --prefix dlib/ https://github.com/konstantinjdobler/dlib.git main --squash
```

Pushing new commits. It's best to not mix changes in a subtree and the host repo in a single commit:

```bash
git subtree push --prefix dlib/ https://github.com/konstantinjdobler/dlib.git main
```

Useful resource: https://gist.github.com/SKempin/b7857a6ff6bddb05717cc17a44091202
