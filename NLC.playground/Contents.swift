
//: Playground - noun: a place where people can play

//import UIKit
import Foundation


typealias TaggedToken = (String, String?)

func tag(text: String, scheme: String) -> [TaggedToken] {
    let options: NSLinguisticTagger.Options = [.omitWhitespace, .omitPunctuation, .omitOther]
    let tagger = NSLinguisticTagger(tagSchemes: NSLinguisticTagger.availableTagSchemes(forLanguage: "en"), options: Int(options.rawValue))
    
    tagger.string = text
    
    var tokens: [TaggedToken] = []
    
    // Using NSLinguisticTagger
    tagger.enumerateTags(in: NSMakeRange(0, text.characters.count), scheme:NSLinguisticTagScheme(rawValue: scheme), options: options) { tag, tokenRange, _, _ in
        let token = (text as NSString).substring(with: tokenRange)
        tokens.append((token, tag?.rawValue))
    }
    
    
    return tokens
}

func partOfSpeech(text: String) -> [TaggedToken] {
    return tag(text: text, scheme: NSLinguisticTagScheme.lexicalClass.rawValue)
}

partOfSpeech(text: "I went to the store")
//partOfSpeech(text: "I am talking quickly")

func lemmatize(text: String) -> [TaggedToken] {
    return tag(text: text, scheme: NSLinguisticTagScheme.lemma.rawValue)
}

lemmatize(text: "I went to the store")


func language(text: String) -> [TaggedToken] {
    return tag(text: text, scheme: NSLinguisticTagScheme.language.rawValue)
}

language(text: "Io vado al negozio")
language(text: "I went to the store")



public class NaiveBayesClassifier {
    public typealias Category = String
    public typealias Tokenizer = (String) -> [String]
    
    private let tokenizer: Tokenizer
    
    private var categoryOccurrences: [Category: Int] = [:]
    private var tokenOccurrences: [String: [Category: Int]] = [:]
    private var trainingCount = 0
    private var tokenCount = 0
    
    private let smoothingParameter = 1.0
    
    public init(tokenizer: @escaping (Tokenizer)) {
        self.tokenizer = tokenizer
    }
    
    // MARK: - Training
    
    public func trainWithText(text: String, category: Category) {
        trainWithTokens(tokens: tokenizer(text), category: category)
    }
    
    public func trainWithTokens(tokens: [String], category: Category) {
        let tokens = Set(tokens)
        for token in tokens {
            incrementToken(token: token, category: category)
        }
        incrementCategory(category: category)
        trainingCount += 1
    }
    
    // MARK: - Classifying
    
    public func classifyText(text: String) -> Category? {
        return classifyTokens(tokens: tokenizer(text))
    }
    
    public func classifyTokens(tokens: [String]) -> Category? {
        // Compute argmax_cat [log(P(C=cat)) + sum_token(log(P(W=token|C=cat)))]
        var maxCategory: Category?
        var maxCategoryScore = -Double.infinity
        for (category, _) in categoryOccurrences {
            let pCategory = P(category: category)
            let score = tokens.reduce(log(pCategory)) { (total, token) in
                // P(W=token|C=cat) = P(C=cat, W=token) / P(C=cat)
                total + log((P(category: category, token) + smoothingParameter) / (pCategory + smoothingParameter * Double(tokenCount)))
            }
            if score > maxCategoryScore {
                maxCategory = category
                maxCategoryScore = score
            }
        }
        return maxCategory
    }
    
    // MARK: - Probabilites
    
    private func P(category: Category, _ token: String) -> Double {
        return Double(tokenOccurrences[token]?[category] ?? 0) / Double(trainingCount)
    }
    
    private func P(category: Category) -> Double {
        return Double(totalOccurrencesOfCategory(category: category)) / Double(trainingCount)
    }
    
    // MARK: - Counting
    
    private func incrementToken(token: String, category: Category) {
        if tokenOccurrences[token] == nil {
            tokenCount += 1
            tokenOccurrences[token] = [:]
        }
        
        // Force unwrap to crash instead of providing faulty results.
        let count = tokenOccurrences[token]![category] ?? 0
        tokenOccurrences[token]![category] = count + 1
    }
    
    private func incrementCategory(category: Category) {
        categoryOccurrences[category] = totalOccurrencesOfCategory(category: category) + 1
    }
    
    private func totalOccurrencesOfToken(token: String) -> Int {
        if let occurrences = tokenOccurrences[token] {
            
            return occurrences.values.reduce(0, +)
            //return occurrences.values.reduce(0) { $0 + $1 }
            //return reduce(occurrences.values, 0, +)
        }
        return 0
    }
    
    private func totalOccurrencesOfCategory(category: Category) -> Int {
        return categoryOccurrences[category] ?? 0
    }
}

let nbc = NaiveBayesClassifier { (text: String) -> [String] in
    return lemmatize(text: text).map { (token, tag) in
        return tag ?? token
    }
}



nbc.trainWithText(text: "spammy spam spam", category: "spam")
nbc.trainWithText(text: "spam has a lot of sodium and cholesterol", category: "spam")

nbc.trainWithText(text: "nom nom ham", category: "ham")
nbc.trainWithText(text: "please put the ham and eggs in the fridge", category: "ham")

nbc.classifyText(text: "sodium and cholesterol")
nbc.classifyText(text: "spam and eggs")
nbc.classifyText(text: "do you like spam?")

nbc.classifyText(text: "use the eggs in the fridge")
nbc.classifyText(text: "ham and eggs")
nbc.classifyText(text: "do you like ham?")

nbc.classifyText(text: "do you eat egg?")

