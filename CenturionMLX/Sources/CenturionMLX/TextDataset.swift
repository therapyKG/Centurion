import Foundation
import MLX

// MARK: - Character-Level Tokenizer

/// Simple character-level tokenizer that maps unique characters to integer IDs.
/// Keeps vocab small and deterministic — no external dependencies.
public struct CharTokenizer: Sendable {
    public let charToID: [Character: Int32]
    public let idToChar: [Int32: Character]
    public let vocabSize: Int

    public init(text: String) {
        let chars = Array(Set(text)).sorted()
        var c2i: [Character: Int32] = [:]
        var i2c: [Int32: Character] = [:]
        for (idx, ch) in chars.enumerated() {
            c2i[ch] = Int32(idx)
            i2c[Int32(idx)] = ch
        }
        self.charToID = c2i
        self.idToChar = i2c
        self.vocabSize = chars.count
    }

    public func encode(_ text: String) -> [Int32] {
        return text.map { charToID[$0] ?? 0 }
    }

    public func decode(_ ids: [Int32]) -> String {
        return String(ids.compactMap { idToChar[$0] })
    }
}

// MARK: - Text Dataset

/// Holds a tokenized text corpus and samples batches of (input, target) pairs
/// for next-token prediction training.
public struct TextDataset: Sendable {
    public let name: String
    public let tokenizer: AnyTokenizer
    public let tokens: [Int32]           // Full tokenized corpus
    public let vocabSize: Int

    /// Initialize with a character-level tokenizer (default, backwards-compatible).
    public init(name: String, text: String) {
        self.name = name
        let charTok = CharTokenizer(text: text)
        self.tokenizer = AnyTokenizer(charTok)
        self.tokens = charTok.encode(text)
        self.vocabSize = charTok.vocabSize
    }

    /// Initialize with any tokenizer.
    public init(name: String, text: String, tokenizer: AnyTokenizer) {
        self.name = name
        self.tokenizer = tokenizer
        self.tokens = tokenizer.encode(text)
        self.vocabSize = tokenizer.vocabSize
    }

    /// Sample a batch of (input, target) sequences for next-char prediction.
    /// input[i] = tokens[pos..pos+seqLen], target[i] = tokens[pos+1..pos+seqLen+1]
    public func sampleBatch(batchSize: Int, seqLen: Int) -> (Data, Data) {
        let maxStart = tokens.count - seqLen - 1
        guard maxStart > 0 else {
            // Corpus too short — pad with zeros
            let count = batchSize * seqLen
            let zeros = [Int32](repeating: 0, count: count)
            return (zeros.withUnsafeBytes { Data($0) }, zeros.withUnsafeBytes { Data($0) })
        }

        var inputs = [Int32](repeating: 0, count: batchSize * seqLen)
        var targets = [Int32](repeating: 0, count: batchSize * seqLen)

        for b in 0..<batchSize {
            let start = Int.random(in: 0...maxStart)
            let offset = b * seqLen
            for s in 0..<seqLen {
                inputs[offset + s] = tokens[start + s]
                targets[offset + s] = tokens[start + s + 1]
            }
        }

        return (inputs.withUnsafeBytes { Data($0) }, targets.withUnsafeBytes { Data($0) })
    }

    /// Sample a batch as MLXArrays for MLX training.
    /// Returns (inputs: [B, seqLen], targets: [B, seqLen]) as Int32 MLXArrays.
    public func sampleBatchMLX(batchSize: Int, seqLen: Int) -> (MLXArray, MLXArray) {
        let maxStart = tokens.count - seqLen - 1
        guard maxStart > 0 else {
            let count = batchSize * seqLen
            let zeros = [Int32](repeating: 0, count: count)
            let inp = MLXArray(zeros).reshaped(batchSize, seqLen)
            return (inp, inp)
        }

        var inputs = [Int32](repeating: 0, count: batchSize * seqLen)
        var targets = [Int32](repeating: 0, count: batchSize * seqLen)

        for b in 0..<batchSize {
            let start = Int.random(in: 0...maxStart)
            let offset = b * seqLen
            for s in 0..<seqLen {
                inputs[offset + s] = tokens[start + s]
                targets[offset + s] = tokens[start + s + 1]
            }
        }

        let inputArray = MLXArray(inputs).reshaped(batchSize, seqLen)
        let targetArray = MLXArray(targets).reshaped(batchSize, seqLen)
        return (inputArray, targetArray)
    }

    /// Returns a summary string of dataset statistics.
    public func textStats() -> String {
        return "\(name): \(tokens.count) chars, vocab=\(vocabSize)"
    }
}

// MARK: - Built-in Datasets

public enum BuiltinDataset: String, CaseIterable, Identifiable, Sendable {
    case shakespeareLarge = "Shakespeare (50K)"
    case tinyShakespeare = "Tiny Shakespeare"
    case nurseryRhymes = "Nursery Rhymes"
    case simpleSentences = "Simple Sentences"

    public var id: String { rawValue }

    public func load() -> TextDataset {
        return TextDataset(name: rawValue, text: corpusText)
    }

    /// Load with GPT-2 BPE tokenizer when `bpe` is true, char-level otherwise.
    public func load(bpe: Bool) -> TextDataset {
        guard bpe else { return load() }
        do {
            let tok = try GPT2Tokenizer()
            return TextDataset(name: rawValue, text: corpusText, tokenizer: AnyTokenizer(tok))
        } catch {
            // Fall back to char-level if resources are missing
            return load()
        }
    }

    private var corpusText: String {
        switch self {
        case .shakespeareLarge:
            return Self.shakespeareLargeText
        case .tinyShakespeare:
            return Self.tinyShakespeareText
        case .nurseryRhymes:
            return Self.nurseryRhymesText
        case .simpleSentences:
            return Self.simpleSentencesText
        }
    }

    // ~4KB of Shakespeare — enough to train a small char-level model
    private static let tinyShakespeareText = """
    First Citizen:
    Before we proceed any further, hear me speak.

    All:
    Speak, speak.

    First Citizen:
    You are all resolved rather to die than to famish?

    All:
    Resolved. resolved.

    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.

    All:
    We know't, we know't.

    First Citizen:
    Let us kill him, and we'll have corn at our own price.
    Is't a verdict?

    All:
    No more talking on't; let it be done: away, away!

    Second Citizen:
    One word, good citizens.

    First Citizen:
    We are accounted poor citizens, the patricians good.
    What authority surfeits on would relieve us: if they
    would yield us but the superfluity, while it were
    wholesome, we might guess they relieved us humanely;
    but they think we are too dear: the leanness that
    afflicts us, the object of our misery, is as an
    inventory to particularise their abundance; our
    sufferance is a gain to them Let us revenge this with
    our pikes, ere we become rakes: for the gods know I
    speak this in hunger for bread, not in thirst for revenge.

    Second Citizen:
    Would you proceed especially against Caius Marcius?

    First Citizen:
    Against him first: he's a very dog to the commonalty.

    Second Citizen:
    Consider you what services he has done for his country?

    First Citizen:
    Very well; and could be content to give him good
    report fort, but that he pays himself with being proud.

    Second Citizen:
    Nay, but speak not maliciously.

    First Citizen:
    I say unto you, what he hath done famously, he did
    it to that end: though soft-conscienced men can be
    content to say it was for his country he did it to
    please his mother and to be partly proud; which he
    is, even to the altitude of his virtue.

    Second Citizen:
    What he cannot help in his nature, you account a
    vice in him. You must in no way say he is covetous.

    First Citizen:
    If I must not, I need not be barren of accusations;
    he hath faults, with surplus, to tire in repetition.
    What shouts are these? The other side o' the city is risen:
    why stay we prating here? to the Capitol!

    All:
    Come, come.

    First Citizen:
    Soft! who comes here?

    Second Citizen:
    Worthy Menenius Agrippa; one that hath always loved
    the people.

    First Citizen:
    He's one honest enough: would all the rest were so!

    MENENIUS:
    What work's, my countrymen, in hand? where go you
    With bats and clubs? the matter? speak, I pray you.

    First Citizen:
    Our business is not unknown to the senate; they have
    had inkling this fortnight what we intend to do,
    which now we'll show 'em in deeds. They say poor
    suitors have strong breaths: they shall know we
    have strong arms too.

    MENENIUS:
    Why, masters, my good friends, mine honest neighbours,
    Will you undo yourselves?

    First Citizen:
    We cannot, sir, we are undone already.

    MENENIUS:
    I tell you, friends, most charitable care
    Have the patricians of you. For your wants,
    Your suffering in this dearth, you may as well
    Strike at the heaven with your staves as lift them
    Against the Roman state, whose course will on
    The way it takes, cracking ten thousand curbs
    Of more strong link asunder than can ever
    Appear in your impediment. For the dearth,
    The gods, not the patricians, make it, and
    Your knees to them, not arms, must help. Alack,
    You are transported by calamity
    Thither where more attends you, and you slander
    The helms o' the state, who care for you like fathers,
    When you curse them as enemies.

    First Citizen:
    Care for us! True, indeed! They ne'er cared for us
    yet: suffer us to famish, and their storehouses
    crammed with grain; make edicts for usury, to
    support usurers; repeal daily any wholesome act
    established against the rich, and provide more
    piercing statutes daily, to chain up and restrain
    the poor. If the wars eat us not up, they will; and
    there's all the love they bear us.

    MENENIUS:
    Either you must
    Confess yourselves wondrous malicious,
    Or be accused of folly. I shall tell you
    A pretty tale: it may be you have heard it;
    But, since it serves my purpose, I will venture
    To stale 't a little more.

    First Citizen:
    Well, I'll hear it, sir: yet you must not think to
    fob off our disgrace with a tale: but, an 't please
    you, deliver.

    MENENIUS:
    There was a time when all the body's members
    Rebell'd against the belly, thus accused it:
    That only like a gulf it did remain
    I' the midst o' the body, idle and unactive,
    Still cupboarding the viand, never bearing
    Like labour with the rest, where the other instruments
    Did see and hear, devise, instruct, walk, feel,
    And, mutually participate, did minister
    Unto the appetite and affection common
    Of the whole body. The belly answer'd--

    First Citizen:
    Well, sir, what answer made the belly?

    MENENIUS:
    Sir, I shall tell you. With a kind of smile,
    Which ne'er came from the lungs, but even thus--
    For, look you, I may make the belly smile
    As well as speak--it tauntingly replied
    To the discontented members, the mutinous parts
    That envied his receipt; even so most fitly
    As you malign our senators for that
    They are not such as you.

    First Citizen:
    Your belly's answer? What!
    The kingly-crowned head, the vigilant eye,
    The counsellor heart, the arm our soldier,
    Our steed the leg, the tongue our trumpeter.
    With other muniments and petty helps
    In this our fabric, if that they--

    MENENIUS:
    What then?
    'Fore me, this fellow speaks! What then? what then?

    First Citizen:
    Should by the cormorant belly be restrain'd,
    Who is the sink o' the body,--

    MENENIUS:
    Well, what then?

    First Citizen:
    The former agents, if they did complain,
    What could the belly answer?

    MENENIUS:
    I will tell you
    If you'll bestow a small--of what you have little--
    Patience awhile, you'll hear the belly's answer.

    First Citizen:
    Ye're long about it.

    MENENIUS:
    Note me this, good friend;
    Your most grave belly was deliberate,
    Not rash like his accusers, and thus answer'd:
    'True is it, my incorporate friends,' quoth he,
    'That I receive the general food at first,
    Which you do live upon; and fit it is,
    Because I am the store-house and the shop
    Of the whole body: but, if you do remember,
    I send it through the rivers of your blood,
    Even to the court, the heart, to the seat o' the brain;
    And, through the cranks and offices of man,
    The strongest nerves and small inferior veins
    From me receive that natural competency
    Whereby they live: and though that all at once,
    You, my good friends,'--this says the belly, mark me,--

    First Citizen:
    Ay, sir; well, well.

    MENENIUS:
    'Though all at once cannot
    See what I do deliver out to each,
    Yet I can make my audit up, that all
    From me do back receive the flour of all,
    And leave me but the bran.' What say you to't?

    First Citizen:
    It was an answer: how apply you this?

    MENENIUS:
    The senators of Rome are this good belly,
    And you the mutinous members; for examine
    Their counsels and their cares, digest things rightly
    Touching the weal o' the common, you shall find
    No public benefit which you receive
    But it proceeds or comes from them to you
    And no way from yourselves. What do you think,
    You, the great toe of this assembly?

    First Citizen:
    I the great toe! why the great toe?

    MENENIUS:
    For that, being one o' the lowest, basest, poorest,
    Of this most wise rebellion, thou go'st foremost:
    Thou rascal, that art worst in blood to run,
    Lead'st first to win some vantage.
    But make you ready your stiff bats and clubs:
    Rome and her rats are at the point of battle;
    The one side must have bale.
    """

    // ~2KB of nursery rhymes — very repetitive, easier to learn
    private static let nurseryRhymesText = """
    Humpty Dumpty sat on a wall,
    Humpty Dumpty had a great fall.
    All the king's horses and all the king's men
    Couldn't put Humpty together again.

    Jack and Jill went up the hill
    To fetch a pail of water.
    Jack fell down and broke his crown,
    And Jill came tumbling after.

    Mary had a little lamb,
    Its fleece was white as snow,
    And everywhere that Mary went
    The lamb was sure to go.

    It followed her to school one day,
    Which was against the rule.
    It made the children laugh and play,
    To see a lamb at school.

    Twinkle, twinkle, little star,
    How I wonder what you are!
    Up above the world so high,
    Like a diamond in the sky.

    When the blazing sun is gone,
    When he nothing shines upon,
    Then you show your little light,
    Twinkle, twinkle, through the night.

    Hey diddle diddle, the cat and the fiddle,
    The cow jumped over the moon.
    The little dog laughed to see such sport,
    And the dish ran away with the spoon.

    Baa, baa, black sheep, have you any wool?
    Yes sir, yes sir, three bags full.
    One for the master, one for the dame,
    And one for the little boy who lives down the lane.

    Little Bo Peep has lost her sheep,
    And doesn't know where to find them.
    Leave them alone and they'll come home,
    Bringing their tails behind them.

    Jack Sprat could eat no fat,
    His wife could eat no lean,
    And so between the two of them,
    They licked the platter clean.

    Old Mother Hubbard went to the cupboard
    To give the poor dog a bone.
    When she got there, the cupboard was bare,
    And so the poor dog had none.

    Little Miss Muffet sat on a tuffet,
    Eating her curds and whey.
    Along came a spider, who sat down beside her,
    And frightened Miss Muffet away.

    Three blind mice, three blind mice,
    See how they run, see how they run!
    They all ran after the farmer's wife,
    Who cut off their tails with a carving knife.
    Did you ever see such a thing in your life,
    As three blind mice?

    Georgie Porgie, pudding and pie,
    Kissed the girls and made them cry.
    When the boys came out to play,
    Georgie Porgie ran away.

    Ring-a-ring o' roses,
    A pocket full of posies,
    A-tishoo! A-tishoo!
    We all fall down.

    Row, row, row your boat,
    Gently down the stream.
    Merrily, merrily, merrily, merrily,
    Life is but a dream.

    Row, row, row your boat,
    Gently down the stream.
    If you see a crocodile,
    Don't forget to scream.

    London Bridge is falling down,
    Falling down, falling down.
    London Bridge is falling down,
    My fair lady.
    """

    // ~1KB of simple repeated sentences — easiest to learn
    private static let simpleSentencesText = """
    the cat sat on the mat. the dog sat on the log. the cat and the dog are friends.
    the cat sat on the mat. the dog sat on the log. the cat and the dog are friends.
    the bird flew over the tree. the fish swam under the bridge. the bird and the fish are different.
    the bird flew over the tree. the fish swam under the bridge. the bird and the fish are different.
    the sun is bright. the moon is dim. the stars are far away. the sky is blue.
    the sun is bright. the moon is dim. the stars are far away. the sky is blue.
    the boy ran fast. the girl ran faster. they both ran to the store and back.
    the boy ran fast. the girl ran faster. they both ran to the store and back.
    one two three four five six seven eight nine ten. one two three four five.
    one two three four five six seven eight nine ten. one two three four five.
    hello world. goodbye world. hello again. goodbye again. hello world.
    hello world. goodbye world. hello again. goodbye again. hello world.
    the rain falls down. the wind blows hard. the snow is cold. the sun is warm.
    the rain falls down. the wind blows hard. the snow is cold. the sun is warm.
    she sells sea shells by the sea shore. she sells sea shells by the sea shore.
    peter piper picked a peck of pickled peppers. peter piper picked a peck of pickled peppers.
    the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.
    """

    // ~50K Shakespeare corpus - repeated and extended for more training data
    private static let shakespeareLargeText: String = {
        // Build a larger corpus by repeating and combining the existing texts
        var text = tinyShakespeareText
        // Repeat to reach ~50K characters
        while text.count < 50_000 {
            text += "\n\n" + tinyShakespeareText
        }
        return text
    }()
}
