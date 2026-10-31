function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

function walk(tree, x) {
  var n = 0;
  while (tree.f[n] >= 0) {
    n = Math.fround(x[tree.f[n]]) <= tree.t[n] ? tree.l[n] : tree.r[n];
  }
  return tree.v[n];
}

var MODEL_FNS = {
  lr: function (m, x) {
    var z = m.intercept;
    for (var i = 0; i < x.length; i++) z += m.coef[i] * (x[i] - m.mean[i]) / m.scale[i];
    return sigmoid(z);
  },
  dt: function (m, x) { return walk(m, x); },
  rf: function (m, x) {
    var s = 0;
    for (var i = 0; i < m.length; i++) s += walk(m[i], x);
    return s / m.length;
  },
  gb: function (m, x) {
    var z = m.init;
    for (var i = 0; i < m.trees.length; i++) z += m.lr * walk(m.trees[i], x);
    return sigmoid(z);
  }
};

var RACE_NAMES = ["Mexican American", "Other Hispanic", "NH White", "NH Black", "NH Asian", "Other/multiracial"];

function featureVector(names, v) {
  var map = {
    "age": v.age,
    "female": v.female,
    "income_poverty_ratio": v.pir,
    "education (1-5)": v.educ,
    "ever_smoked": v.smoked,
    "alcohol (0-2)": v.alcohol,
    "leisure_activity_min_wk": v.activity,
    "sleep_hours": v.sleep,
    "bmi": v.bmi,
    "phq9": v.phq9
  };
  for (var i = 0; i < RACE_NAMES.length; i++) map["race: " + RACE_NAMES[i]] = (v.race === RACE_NAMES[i]) ? 1 : 0;
  return names.map(function (n) {
    if (!(n in map)) throw new Error("Unknown feature " + n);
    return map[n];
  });
}

function predictAll(bundle, outcome, usePhq, v) {
  var set = usePhq ? "phq" : "base";
  var models = bundle.outcomes[outcome].models[set];
  var x = featureVector(bundle.features[set], v);
  var out = {};
  Object.keys(MODEL_FNS).forEach(function (k) { out[k] = MODEL_FNS[k](models[k], x); });
  return out;
}


function lrContributions(bundle, outcome, usePhq, v) {
  var set = usePhq ? "phq" : "base";
  var m = bundle.outcomes[outcome].models[set].lr, names = bundle.features[set];
  var x = featureVector(names, v), out = {};
  names.forEach(function (n, i) {
    var c = m.coef[i] * (x[i] - m.mean[i]) / m.scale[i];
    var key = n.indexOf("race: ") === 0 ? "race" : n;
    out[key] = (out[key] || 0) + c;
  });
  return out;
}

if (typeof module !== "undefined") module.exports = { predictAll: predictAll, featureVector: featureVector, lrContributions: lrContributions };
