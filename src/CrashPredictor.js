// Crash Predictor — AI-версия на базе React и Firebase
import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import { initializeApp } from "firebase/app";
import { getFirestore, collection, addDoc, getDocs } from "firebase/firestore";
import { getAuth, onAuthStateChanged, signInWithEmailAndPassword, signOut } from "firebase/auth";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const firebaseConfig = {
  apiKey: "AIzaSyDao_tTRmfyXmmE2pQ9SyMVoF7BrmSUHcc",
  authDomain: "majpred.firebaseapp.com",
  projectId: "majpred",
  storageBucket: "majpred.firebasestorage.app",
  messagingSenderId: "143505491426",
  appId: "1:143505491426:web:213f6019ee3993a83a65bb",
  measurementId: "G-ZT2BKWCB0P"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);
const auth = getAuth(app);

export default function CrashPredictor() {
  const [user, setUser] = useState(null);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [crashHistory, setCrashHistory] = useState([]);
  const [model, setModel] = useState(null);
  const [nextPrediction, setNextPrediction] = useState(null);

  useEffect(() => {
    onAuthStateChanged(auth, (currentUser) => setUser(currentUser));
  }, []);

  useEffect(() => {
    if (user) loadModel();
  }, [user]);

  const loadModel = async () => {
    try {
      const loaded = await tf.loadLayersModel("indexeddb://crash-predictor-model");
      loaded.compile({ optimizer: "adam", loss: "meanSquaredError" });
      setModel(loaded);
    } catch {
      const input = tf.input({ shape: [10] });
      const dense1 = tf.layers.dense({ units: 32, activation: "relu" }).apply(input);
      const dense2 = tf.layers.dense({ units: 1, activation: "linear" }).apply(dense1);
      const net = tf.model({ inputs: input, outputs: dense2 });
      net.compile({ optimizer: "adam", loss: "meanSquaredError" });
      setModel(net);
    }
  };

  const saveToFirestore = async (value) => {
    if (user) await addDoc(collection(db, "crash_sessions"), { uid: user.uid, crash: value, timestamp: new Date() });
  };

  const addCrash = async (value) => {
    const newHistory = [...crashHistory, value];
    setCrashHistory(newHistory);
    await saveToFirestore(value);

    if (model && newHistory.length >= 10) {
      const input = newHistory.slice(-10);
      const xs = tf.tensor2d([input]);
      const ys = tf.tensor2d([[value]]);
      await model.fit(xs, ys, { epochs: 3 });
      await model.save("indexeddb://crash-predictor-model");
      predictForward(newHistory);
    }
  };

  const predictForward = async (history) => {
    if (!model || history.length < 10) return;
    const input = history.slice(-10);
    const xs = tf.tensor2d([input]);
    const prediction = await model.predict(xs).data();
    setNextPrediction(prediction[0]);
  };

  const trainFromAll = async () => {
    const snapshot = await getDocs(collection(db, "crash_sessions"));
    const data = snapshot.docs.map(doc => doc.data().crash);
    if (data.length >= 11 && model) {
      const inputs = [];
      const outputs = [];
      for (let i = 10; i < data.length; i++) {
        inputs.push(data.slice(i-10, i));
        outputs.push(data[i]);
      }
      const xs = tf.tensor2d(inputs);
      const ys = tf.tensor2d(outputs.map(v => [v]));
      await model.fit(xs, ys, { epochs: 5 });
      await model.save("indexeddb://crash-predictor-model");
      alert("Модель обучена на всех данных!");
    }
  };

  const signIn = () => signInWithEmailAndPassword(auth, email, password).catch(console.error);
  const signOutUser = () => signOut(auth);

  if (!user) return (
    <div style={{ padding: 20 }}>
      <h2>🔐 Вход</h2>
      <input placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} />
      <input placeholder="Пароль" type="password" value={password} onChange={e => setPassword(e.target.value)} />
      <button onClick={signIn}>Войти</button>
    </div>
  );

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>🚀 Crash Predictor AI</h1>
      <p>👤 Пользователь: {user.email} <button onClick={signOutUser}>Выйти</button></p>
      <div style={{ marginBottom: 20 }}>
        <input type="number" placeholder="Введите коэффициент краша (например, 1.35)"
          onKeyDown={e => e.key === 'Enter' && addCrash(parseFloat(e.target.value))}
          style={{ padding: '8px', marginRight: '10px' }} />
        <button onClick={trainFromAll}>📚 Обучить на всех данных</button>
      </div>
      {nextPrediction && (
        <div style={{ marginBottom: 20 }}>
          <h2>🔮 Следующий прогноз коэффициента: {nextPrediction.toFixed(2)}x</h2>
        </div>
      )}
      <h2>📈 История крашей</h2>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={crashHistory.map((c, i) => ({ name: i+1, value: c }))}>
          <XAxis dataKey="name" />
          <YAxis domain={[0, 'auto']} />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
