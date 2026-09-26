# CATT's markers, measured

What the diacritizer's non-standard characters actually encode, established by
running the ECA checkpoint on probe sentences rather than by assumption. These
measurements decide which markers become phonemic features and which are deleted.

---

## The triple dot (`؞`, U+061E) is phonemic

It attaches to exactly three letters and nothing else:

| Letter | Count in probe | Egyptian effect |
|---|---|---|
| `ق` | 5 | qaf → hamza (glottal stop) |
| `ج` | 1 | /g/ → /ʒ/ (geem → jeem) |
| `ف` | 1 | /f/ → /v/ |

Real output: `إِلتُّكْنُوْلُوْج؞ْيَا` (technology, /ʒ/), `دِلْوَق؞ْتِي` (now, hamza),
`إِلْف؞ِيْدْيُو` (video, /v/).

All three are the same phenomenon: a **non-native or colloquial realisation** of a
letter whose default reading differs. Egyptian /g/ for ج is the default, so a
loanword needing /ʒ/ is marked. Egyptian hamza for ق is the colloquial default,
so it is marked where MSA spelling would suggest qaf.

### It is a strong signal, not a rule

Measured on twelve words with a known Egyptian pronunciation:

| | marked | unmarked |
|---|---|---|
| hamza (colloquial register) | **5** | 1 |
| qaf (formal/MSA register) | 1 | **5** |

83% agreement. The errors are real: `ق؞َرَارْ` (qarar, "decision") is marked
although it keeps its qaf, and `اُقعُدْ` (u'ud, "sit") is unmarked although it
takes hamza. The same word can also differ between runs in different contexts:
`قررت` came back marked in one sentence and unmarked in another.

**Consequence for the design.** The marker is good enough to be a *feature the
acoustic model learns from*, alongside the audio, which can overrule it. It is
not good enough to be a hard grapheme-to-phoneme rule, and it is not good enough
to define separate pronunciation codes on its own: doing that would split قرار
into two readings on a diacritizer error, which is the failure that cost the
first run.

So the triple dot is preserved as a **phoneme-variant feature** on its letter,
never discarded, and never promoted to a reading by itself.

---

## The caret (`^`) and angle brackets are noise

The caret attaches to anything: measured on و ع م ر ز ن ه د, with no phonological
pattern. It appears *between a letter and its mark* (`م^َصْرِ`, `الن^َّظَرِي^َّه`),
which is why it has to be removed before marks are bound to letters rather than
after. Deleted, along with `<`, `>`, `` ` `` and the ellipsis.

---

## Final `ه` vs `ة` and the /t/ contrast

The corpus spells these interchangeably: `مدينة` and `مدينه` are the same word.
CATT diacritizes the taa-marbuta spelling correctly and the heh spelling
inconsistently, so the *spelling* difference must not create a reading.

But one real contrast lives here:

| Form | Pronunciation |
|---|---|
| `مَدِينَة` (mark, no sukun) | /madiina/ — the taa is silent |
| `مَدِينَةْ` (sukun on the taa-marbuta) | /madiinat/ — the taa is pronounced |

Sukun on a word-final `ة` means the /t/ surfaces. That is phonemic and is kept as
a feature. A mark that is *not* sukun on a final `ه`/`ة` is a case ending, which
Egyptian drops, and is suppressed.

The distinction matters because the two rules pull in opposite directions:
suppressing case endings must not also suppress the /t/ signal.
